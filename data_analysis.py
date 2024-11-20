import os
import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging


class CorpusAnalyzer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.brazilian_corpus_path = self.base_path / "Brazilian_Portugese_Corpus"
        self.machado_path = self.base_path / "machado_de_assis/raw/txt"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # # Download required NLTK data
        # nltk.download('punkt')
        # nltk.download('stopwords')

        self.portuguese_stopwords = set(
            nltk.corpus.stopwords.words('portuguese'))

    def read_file(self, file_path: Path) -> str:
        """Read a text file and return its content."""
        try:
            # Try different encodings commonly used for Portuguese text
            encodings = ['latin1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            self.logger.error(
                f"Failed to read {file_path} with any supported encoding")
            return ""
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {str(e)}")
            return ""

    def get_text_stats(self, text: str) -> Dict[str, int]:
        """Calculate basic statistics for a text."""
        sentences = sent_tokenize(text.lower(), language='portuguese')
        words = word_tokenize(text.lower(), language='portuguese')
        # Remove non-alphanumeric words but don't filter stopwords
        words_filtered = [w for w in words if w.isalnum()]

        # Calculate word lengths
        word_lengths = [len(word) for word in words_filtered]

        # Calculate lexical density (content words / total words)
        content_words = [
            w for w in words_filtered if w not in self.portuguese_stopwords]
        lexical_density = len(content_words) / \
            len(words_filtered) if words_filtered else 0

        # Calculate type-token ratio (TTR)
        ttr = len(set(words_filtered)) / \
            len(words_filtered) if words_filtered else 0

        # Calculate hapax legomena (words that appear only once)
        word_freq = defaultdict(int)
        for word in words_filtered:
            word_freq[word] += 1
        hapax = len([word for word, freq in word_freq.items() if freq == 1])

        return {
            'num_chars': len(text),
            'num_words': len(words),
            'num_unique_words': len(set(words_filtered)),
            'num_sentences': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': sum(word_lengths) / len(word_lengths) if word_lengths else 0,
            'lexical_density': lexical_density,
            'type_token_ratio': ttr,
            'hapax_legomena': hapax,
            'hapax_percentage': (hapax / len(words_filtered) * 100) if words_filtered else 0
        }

    def analyze_corpus(self) -> Tuple[pd.DataFrame, Dict]:
        """Analyze both corpora and return statistics."""
        stats_data = []
        corpus_stats = {
            'machado': defaultdict(int),
            'brazilian': defaultdict(int)
        }

        # Process Machado corpus
        for category in os.listdir(self.machado_path):
            category_path = self.machado_path / category
            if category_path.is_dir():
                for file_path in category_path.glob('*.txt'):
                    text = self.read_file(file_path)
                    stats = self.get_text_stats(text)
                    stats.update({
                        'author': 'Machado de Assis',
                        'category': category,
                        'is_machado': 1,
                        'file_name': file_path.name
                    })
                    stats_data.append(stats)

                    # Update corpus stats
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            corpus_stats['machado'][key] += value

        # Process Brazilian corpus
        for file_path in self.brazilian_corpus_path.rglob('*.txt'):
            if file_path.name == "guideToDocuments.csv":
                continue

            text = self.read_file(file_path)
            stats = self.get_text_stats(text)

            stats.update({
                'author': 'outros',
                'category': 'general',
                'is_machado': 0,
                'file_name': file_path.name
            })
            stats_data.append(stats)

            # Update corpus stats
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    corpus_stats['brazilian'][key] += value

        return pd.DataFrame(stats_data), corpus_stats

    def plot_statistics(self, df: pd.DataFrame, output_dir: str = "plots"):
        """Generate and save visualization plots."""
        os.makedirs(output_dir, exist_ok=True)

        # Use default style instead of seaborn
        plt.style.use('default')

        # 1. Words per author boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='author', y='num_words')
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribuição de Palavras por Autor')
        plt.xlabel('Autor')
        plt.ylabel('Número de Palavras')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/words_by_author.png")
        plt.close()

        # 2. Average sentence length comparison
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='is_machado', y='avg_sentence_length')
        plt.title('Comprimento Médio das Frases: Machado vs Outros')
        plt.xlabel('Autor')
        plt.ylabel('Comprimento Médio das Frases')
        plt.xticks([0, 1], ['Outros Autores', 'Machado'])
        plt.savefig(f"{output_dir}/avg_sentence_length.png")
        plt.close()

        # 3. Unique words ratio
        df['unique_words_ratio'] = df['num_unique_words'] / df['num_words']
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='is_machado', y='unique_words_ratio')
        plt.title('Riqueza de Vocabulário: Proporção de Palavras Únicas')
        plt.xlabel('Autor')
        plt.ylabel('Proporção de Palavras Únicas')
        plt.xticks([0, 1], ['Outros Autores', 'Machado'])
        plt.savefig(f"{output_dir}/vocabulary_richness.png")
        plt.close()

        # 4. Lexical Density Comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='is_machado', y='lexical_density')
        plt.title('Densidade Lexical: Proporção de Palavras de Conteúdo')
        plt.xlabel('Autor')
        plt.ylabel('Densidade Lexical')
        plt.xticks([0, 1], ['Outros Autores', 'Machado'])
        plt.savefig(f"{output_dir}/lexical_density.png")
        plt.close()

        # 5. Average Word Length
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='is_machado', y='avg_word_length')
        plt.title('Distribuição do Comprimento Médio das Palavras')
        plt.xlabel('Autor')
        plt.ylabel('Comprimento Médio das Palavras')
        plt.xticks([0, 1], ['Outros Autores', 'Machado'])
        plt.savefig(f"{output_dir}/avg_word_length.png")
        plt.close()

        # 6. Hapax Percentage
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='is_machado', y='hapax_percentage')
        plt.title('Porcentagem de Hapax Legomena')
        plt.xlabel('Autor')
        plt.ylabel('Porcentagem')
        plt.xticks([0, 1], ['Outros Autores', 'Machado'])
        plt.savefig(f"{output_dir}/hapax_percentage.png")
        plt.close()


def main():
    analyzer = CorpusAnalyzer("data")
    df, corpus_stats = analyzer.analyze_corpus()

    # Save detailed statistics
    df.to_csv("data/processed/corpus_statistics.csv", index=False)

    # Print summary statistics
    logging.info("\n=== Corpus Statistics ===")
    logging.info("\nMachado de Assis Corpus:")
    for key, value in corpus_stats['machado'].items():
        logging.info(f"{key}: {value:,}")

    logging.info("\nBrazilian Corpus:")
    for key, value in corpus_stats['brazilian'].items():
        logging.info(f"{key}: {value:,}")

    # Generate plots
    analyzer.plot_statistics(df)

    # Additional summary statistics
    logging.info("\n=== Summary by Author ===")
    summary = df.groupby('author').agg({
        'num_words': ['mean', 'std', 'count'],
        'num_unique_words': 'mean',
        'avg_sentence_length': 'mean'
    }).round(2)
    logging.info("\n" + str(summary))


if __name__ == "__main__":
    main()
