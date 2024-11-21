import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np

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


def evaluate_model(path_test_dataset):
    # Load test data
    logger.info("Loading test dataset...")
    test_df = pd.read_csv(path_test_dataset)

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model_path = 'models/machado_classifier'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Create test dataset and dataloader
    test_dataset = MachadoDataset(
        test_df['text'].tolist(),
        test_df['is_machado'].tolist(),
        tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Lists to store predictions and true labels
    all_predictions = []
    all_labels = []

    # Evaluation loop
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    logger.info("Calculating metrics...")
    report = classification_report(all_labels, all_predictions, target_names=[
                                   'Not Machado', 'Machado'])
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Print results
    logger.info("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Machado', 'Machado'],
                yticklabels=['Not Machado', 'Machado'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation_results/confusion_matrix.png')
    plt.close()

    # Save detailed results
    with open('evaluation_results/evaluation_report.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))


if __name__ == "__main__":
    evaluate_model()
