import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str = "data/processed/unified_corpus_chunks.csv") -> pd.DataFrame:
    """Load the unified corpus dataset."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def create_splits(df: pd.DataFrame, random_state: int = 42):
    """
    Create splits for both Machado and non-Machado texts:
    - Machado: 70% train, 30% test
    - Non-Machado: 
        - Test: Same size as Machado test, stratified by source_file
        - Supervised: Same size as Machado train, stratified by source_file
        - Unsupervised: Remaining data, only text column
    All splits are stratified by source_file
    """
    # Separate Machado and non-Machado datasets
    machado_df = df[df['is_machado'] == 1]
    non_machado_df = df[df['is_machado'] == 0]
    
    # Create splits for Machado texts (70% train, 30% test)
    splits_machado = {}
    for source, group in machado_df.groupby('source_file'):
        test = group.sample(frac=0.3, random_state=random_state)
        train = group[~group.index.isin(test.index)]
        
        splits_machado.setdefault('test', []).append(test)
        splits_machado.setdefault('train', []).append(train)
    
    # Combine the splits
    machado_test = pd.concat(splits_machado['test'])
    machado_train = pd.concat(splits_machado['train'])
    
    # Calculate sizes for non-Machado splits
    test_size = len(machado_test)
    train_size = len(machado_train)
    
    # Create non-Machado splits
    splits_non_machado = {'test': [], 'train': [], 'unsupervised': []}
    
    for source, group in non_machado_df.groupby('source_file'):
        # Calculate proportional sizes for this source
        source_fraction = len(group) / len(non_machado_df)
        source_test_size = int(test_size * source_fraction)
        source_train_size = int(train_size * source_fraction)
        
        # Sample for test
        test = group.sample(n=min(source_test_size, len(group)), random_state=random_state)
        remaining = group[~group.index.isin(test.index)]
        
        # Sample for supervised training
        train = remaining.sample(n=min(source_train_size, len(remaining)), random_state=random_state)
        
        # Rest goes to unsupervised
        unsupervised = remaining[~remaining.index.isin(train.index)][['text']]
        
        splits_non_machado['test'].append(test)
        splits_non_machado['train'].append(train)
        splits_non_machado['unsupervised'].append(unsupervised)

    non_machado_supervised_dataset = pd.concat(splits_non_machado['train'])
    supervised_dataset = pd.concat([machado_train, non_machado_supervised_dataset])

    non_machado_test_dataset = pd.concat(splits_non_machado['test'])
    test_dataset = pd.concat([machado_test, non_machado_test_dataset])

    unsupervised_dataset = pd.concat(splits_non_machado['unsupervised'])

    
    # Combine all splits into final dictionary
    final_splits = {
        'supervised': supervised_dataset,
        'test': test_dataset,
        'unsupervised': unsupervised_dataset
    }
    
    # Log split sizes
    logger.info("Split sizes:")
    for split_name, split_df in final_splits.items():
        logger.info(f"{split_name}: {len(split_df)} samples")
    
    return final_splits

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