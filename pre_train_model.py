import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling,
    BertForPreTraining,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    WandbCallback
)
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
import random
import evaluate
import logging
import torch
import nltk
import wandb
from datasets import Dataset, DatasetDict

nltk.download('punkt')  # Descargar los recursos necesarios

RANDOM_SEED = 42

BLOCK_SIZE = 256  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP ##=> Segun BERT
# (10% de las veces vamos a usar secuencias cortas) Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
SHORT_SEQ_PROB = 0.1
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.15  # Probability with which tokens are masked in MLM ##=> BERTimbau 0.15

# Entrenaniento
TRAIN_BATCH_SIZE = 64  # Batch-size for pretraining the model on ##=> BERTimbau base 128
EVAL_BATCH_SIZE = 64
MAX_EPOCHS = 40  # Maximum number of epochs to train the model for
# 2e-5  # Learning rate for training the model ##=> BERTimbau base 1e-4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

# Early stopping
ES_PATIENCE = 2
ES_THRESHOLD = 0.001

PATH_DATASET = ""
PATH_RESULT_MODEL = ""

# Modelo
# Name of pretrained model from 🤗 Model Hub
MODEL_CHECKPOINT = "neuralmind/bert-base-portuguese-cased"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_model(model_name="neuralmind/bert-base-portuguese-cased", num_unfrozen_layers=4):
    # Load pre-trained model and tokenizer
    model = BertForPreTraining.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Freeze layers according to specification
    modules_to_freeze = model.bert.encoder.layer[:-num_unfrozen_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    logger.info(
        f"Model loaded. Frozen all layers except last {num_unfrozen_layers}")
    return model, tokenizer


def prepare_train_features(examples, tokenizer):

    #######
    # NSP
    # We define the maximum number of tokens after tokenization that each training sample
    # will have
    max_num_tokens = BLOCK_SIZE - \
        tokenizer.num_special_tokens_to_add(pair=True)

    """Function to prepare features for NSP task

    Arguments:
      examples: A dictionary with 1 key ("text")
        text: List of raw documents (str)
    Returns:
      examples:  A dictionary with 4 keys
        input_ids: List of tokenized, concatnated, and batched
          sentences from the individual raw documents (int)
        token_type_ids: List of integers (0 or 1) corresponding
          to: 0 for senetence no. 1 and padding, 1 for sentence
          no. 2
        attention_mask: List of integers (0 or 1) corresponding
          to: 1 for non-padded tokens, 0 for padded
        next_sentence_label: List of integers (0 or 1) corresponding
          to: 1 if the second sentence actually follows the first,
          0 if the senetence is sampled from somewhere else in the corpus
    """

    # Remove un-wanted samples from the training set
    examples["document"] = [
        d.strip() for d in examples["text"] if len(d) > 0
    ]
    # Split the documents from the dataset into it's individual sentences
    examples["sentences"] = [
        nltk.tokenize.sent_tokenize(document) for document in examples["document"]
    ]
    # Convert the tokens into ids using the trained tokenizer
    examples["tokenized_sentences"] = [
        [tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sent)) for sent in doc]
        for doc in examples["sentences"]
    ]

    # Define the outputs
    examples["input_ids"] = []
    examples["token_type_ids"] = []
    examples["attention_mask"] = []
    examples["next_sentence_label"] = []

    for doc_index, document in enumerate(examples["tokenized_sentences"]):

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens

        if random.random() < SHORT_SEQ_PROB:
            target_seq_length = random.randint(2, max_num_tokens)

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < NSP_PROB:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(
                                0, len(examples["tokenized_sentences"]) - 1
                            )
                            if random_document_index != doc_index:
                                break

                        random_document = examples["tokenized_sentences"][
                            random_document_index
                        ]
                        random_start = random.randint(
                            0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    # Controlar que los tokens no exedan 512
                    if len(tokens_a) > max_num_tokens:
                        tokens_a = tokens_a[0:max_num_tokens]
                    if len(tokens_b) > max_num_tokens:
                        tokens_b = tokens_b[0:max_num_tokens]

                    input_ids = tokenizer.build_inputs_with_special_tokens(
                        tokens_a, tokens_b
                    )
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
                        tokens_a, tokens_b
                    )

                    padded = tokenizer.pad(
                        {"input_ids": input_ids, "token_type_ids": token_type_ids},
                        padding="max_length",
                        max_length=MAX_LENGTH,
                    )

                    examples["input_ids"].append(padded["input_ids"])
                    examples["token_type_ids"].append(padded["token_type_ids"])
                    examples["attention_mask"].append(padded["attention_mask"])
                    examples["next_sentence_label"].append(
                        1 if is_random_next else 0)
                    current_chunk = []
                    current_length = 0
            i += 1

    # We delete all the un-necessary columns from our dataset
    del examples["document"]
    del examples["sentences"]
    del examples["text"]
    del examples["tokenized_sentences"]

    return examples


def prepare_dataset(tokenizer, csv_path="unsupervised.csv"):
    # Load dataset from CSV
    df = pd.read_csv(csv_path)

    
    # Create DataFrame with only the 'text' column
    x = df['text'].copy()
    
    # Create train/validation split (80/20)
    x_train, x_validation = train_test_split(x, test_size=0.2, random_state=RANDOM_SEED)
    
    # Reset indices
    x_train = x_train.reset_index(drop=True)
    x_validation = x_validation.reset_index(drop=True)
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(x_train, columns=['text']))
    validation_dataset = Dataset.from_pandas(pd.DataFrame(x_validation, columns=['text']))
    
    # Create DatasetDict
    dataset_dict = DatasetDict({'train': train_dataset, 'validation': validation_dataset})
    
    # Tokenize dataset
    tokenized_dataset = dataset_dict.map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        num_proc=12
    )

    logger.info(f"Dataset prepared with NSP features. Using text from {csv_path}")
    return tokenized_dataset


metric = evaluate.load("accuracy")


def preprocess_logits_for_metrics(logits, labels):
    mlm_logits = logits[0]
    nsp_logits = logits[1]
    return mlm_logits.argmax(-1), nsp_logits.argmax(-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    mlm_preds = preds[0]
    nsp_preds = preds[1]

    mlm_labels = labels[0]
    nsp_labels = labels[1]

    mask = mlm_labels != -100
    mlm_labels = mlm_labels[mask]
    mlm_preds = mlm_preds[mask]

    mlm_accuracy = metric.compute(
        predictions=mlm_preds, references=mlm_labels)["accuracy"]
    nsp_accuracy = metric.compute(
        predictions=nsp_preds, references=nsp_labels)["accuracy"]

    return {"Masked ML Accuracy": mlm_accuracy, "NSP Accuracy": nsp_accuracy}


def plot_losses(trainer):
    log_history = trainer.state.log_history
    df_history = pd.DataFrame(log_history)
    set_train_loss = []
    set_val_loss = []
    epochs = int(df_history['epoch'].max())
    for i in range(1, epochs+1):
        index_val = df_history.loc[(
            df_history['epoch'] == i), 'loss'].index.tolist()[0]
        index_train = index_val-1
        train_loss = df_history.loc[index_train, 'loss']
        val_loss = df_history.loc[index_val, 'eval_loss']
        set_train_loss.append(train_loss)
        set_val_loss.append(val_loss)

    # Plot
    loss_index = range(1, len(set_train_loss) + 1)

    plt.figure(figsize=(20, 5))
    plt.plot(loss_index, set_train_loss, label='Train loss')
    plt.plot(loss_index, set_val_loss, label='Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, len(set_train_loss) + 1))
    plt.legend()
    plt.show()


def main():
    # Initialize wandb with updated config
    wandb.init(
        project="bert-pretraining",
        name="bert-pretrain-run",
        config={
            "learning_rate": 1e-4,
            "epochs": 40,
            "batch_size": 64,
            "unfrozen_layers": 4,
            "weight_decay": 0.01
        }
    )

    # Initialize model and tokenizer
    model, tokenizer = load_and_prepare_model(num_unfrozen_layers=4)

    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer)

    # Initialize data collator with NSP
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB
    )

    training_args = TrainingArguments(
        output_dir=PATH_RESULT_MODEL,
        logging_first_step=True,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        fp16=True,
        save_strategy="epoch",
        num_train_epochs=MAX_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience = ES_PATIENCE, early_stopping_threshold = ES_THRESHOLD) 

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stop]
    )
    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    model.save_pretrained("./bert_pretrained_final")
    tokenizer.save_pretrained("./bert_pretrained_final")
    logger.info("Training completed and model saved")

    # Close wandb run when done
    wandb.finish()


if __name__ == "__main__":
    main()
