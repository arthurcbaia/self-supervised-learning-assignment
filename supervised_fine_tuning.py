import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MachadoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_model():
    # Load data
    logger.info("Loading dataset...")
    df = pd.read_csv('data/processed/unified_corpus_chunks.csv')

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['is_machado'].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        'neuralmind/bert-base-portuguese-cased')
    model = BertForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased',
        num_labels=2
    )

    # Create datasets
    train_dataset = MachadoDataset(train_texts, train_labels, tokenizer)
    val_dataset = MachadoDataset(val_texts, val_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3

    # Training loop
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the model
    logger.info("Saving model...")
    model.save_pretrained('models/machado_classifier')
    tokenizer.save_pretrained('models/machado_classifier')


if __name__ == "__main__":
    train_model()
