import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any
import numpy as np

class ChunkAnalyzer:
    def __init__(self, processed_data_path: str):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load the processed data
        self.df = pd.read_csv(processed_data_path)
        self.df['word_count'] = self.df['text'].str.split().str.len()
        
        # Create output directory
        self.output_dir = Path("plots/chunk_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_chunk_distributions(self):
        """Generate distribution plots for chunk analysis."""
        # Set a default style instead of seaborn
        plt.style.use('default')
        
        # Distribution of is_machado
        plt.figure(figsize=(10, 6))
        machado_counts = self.df['is_machado'].value_counts()
        sns.barplot(x=machado_counts.index, y=machado_counts.values)
        plt.title('Distribuição de Chunks por Categoria (Machado vs Outros)')
        plt.xlabel('Categoria')
        plt.ylabel('Número de Chunks')
        plt.xticks([0, 1], ['Outros Autores', 'Machado'])
        # Add value labels on top of each bar
        for i, v in enumerate(machado_counts.values):
            plt.text(i, v, f'{v:,}', ha='center', va='bottom')
        plt.savefig(self.output_dir / "machado_distribution.png")
        plt.close()


        plt.figure(figsize=(12, 6))
        author_counts = self.df['author'].value_counts()
        author_counts.plot(kind='bar')
        plt.title('Número de Chunks por Autor')
        plt.xlabel('Autor')
        plt.ylabel('Número de Chunks')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "chunks_per_author.png")
        plt.close()


    def generate_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the chunks."""
        stats = {
            'total_chunks': len(self.df),
            'machado_chunks': len(self.df[self.df['is_machado'] == 1]),
            'other_chunks': len(self.df[self.df['is_machado'] == 0]),
            'unique_authors': self.df['author'].nunique(),
            'avg_words_per_chunk': self.df['word_count'].mean(),
            'std_words_per_chunk': self.df['word_count'].std(),
            'median_words_per_chunk': self.df['word_count'].median(),
            'min_words': self.df['word_count'].min(),
            'max_words': self.df['word_count'].max()
        }
        
        # Add author-specific statistics
        author_stats = self.df.groupby('author').agg({
            'word_count': ['count', 'mean', 'std'],
            'chunk_id': 'max'
        }).round(2)
        
        return stats, author_stats

def main():
    analyzer = ChunkAnalyzer("data/processed/unified_corpus_chunks.csv")
    
    # Generate and save plots
    analyzer.plot_chunk_distributions()
    
    # Generate and log statistics
    stats, author_stats = analyzer.generate_statistics()
    
    logging.info("\n=== Chunk Analysis Statistics ===")
    for key, value in stats.items():
        logging.info(f"{key}: {value:,.2f}")
    
    logging.info("\n=== Author-specific Statistics ===")
    logging.info("\n" + str(author_stats))

if __name__ == "__main__":
    main() 