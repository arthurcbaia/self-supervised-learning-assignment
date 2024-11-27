import pandas as pd
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

path = 'data/processed/splits/unsupervised.csv'

try:
    logger.info(f"Reading CSV file from {path}")
    df = pd.read_csv(path)
    
    # Calculate distribution for source_file column only
    distributions = {}
    column = 'source_file'
    logger.info(f"Analyzing column: {column}")
    
    # Get value counts and their percentages
    value_counts = df[column].value_counts()
    total_rows = len(df)
    
    distributions[column] = {
        'unique_values': len(value_counts),
        'total_rows': total_rows,
        'most_common': value_counts.to_dict(),  # Get all values, not just top 10
        'repetition_rate': (1 - len(value_counts) / total_rows) * 100
    }
    
    logger.info(f"Column '{column}' has {len(value_counts)} unique values out of {total_rows} rows")

    # Save to JSON file
    output_path = 'data/processed/source_file_distribution.json'
    logger.info(f"Saving source file analysis to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(distributions, f, indent=4)
    
    logger.info("Source file analysis completed successfully")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise
    