import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str = "data/processed/unified_corpus_chunks.csv") -> pd.DataFrame:
    """Load the unified corpus chunks dataset."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def create_splits(df: pd.DataFrame, test_size: float = 0.1, random_state: int = 42):
    """
    Create train/validation/test splits suitable for all training configurations.
    First creates a balanced test set, then splits remaining data into train/valid.
    """
    # First create balanced test set
    machado_test = df[df['is_machado'] == 1].sample(
        frac=test_size, random_state=random_state)
    non_machado_pool = df[df['is_machado'] == 0]

    # Sample same number of non-Machado texts for test set
    non_machado_test = non_machado_pool.sample(
        n=len(machado_test), random_state=random_state)
    test_df = pd.concat([machado_test, non_machado_test])

    # Remove test samples from original dataframe
    train_df = df[~df.index.isin(test_df.index)]

    machado_train = train_df[train_df['is_machado'] == 1]

    # 50% to supervised, 50% to unsupervised
    supervised_df = machado_train.sample(frac=0.5, random_state=random_state)
    unsupervised_df = machado_train[~machado_train.index.isin(
        supervised_df.index)]

    # get sample of non-Machado texts for supervised, with same number of samples as supervised
    supervised_non_machado_df = train_df[train_df['is_machado'] == 0].sample(
        n=len(supervised_df), random_state=random_state)

    supervised_df = pd.concat([supervised_df, supervised_non_machado_df])

    return {
        'test': test_df,
        'supervised': supervised_df,
        'unsupervised': unsupervised_df
    }


def save_splits(splits: dict, output_dir: str = "data/processed/splits"):
    """Save the splits to CSV files."""
    import os

    # Create main splits directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split to a CSV file
    for split_name, split_df in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        logger.info(f"Saving {split_name} split ({len(split_df)} samples) to {output_path}")
        split_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Load data
    df = load_data()

    # Create splits
    splits = create_splits(df)

    # Save splits
    save_splits(splits)
