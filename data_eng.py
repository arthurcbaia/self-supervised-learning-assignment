import os
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re

nltk.download('punkt_tab')


class CorpusProcessor:
    def __init__(self, base_path: str, chunk_size: int = 256):
        """Initialize the corpus processor with the base directory path."""
        self.base_path = Path(base_path)
        self.brazilian_corpus_path = self.base_path / "Brazilian_Portugese_Corpus"
        self.machado_path = self.base_path / "machado_de_assis/raw/txt"
        self.chunk_size = chunk_size

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
        except Exception as e:
            self.logger.error(f"Error downloading NLTK data: {str(e)}")

        # Add more detailed directory validation
        self.logger.info(f"Base path: {self.base_path}")
        self.logger.info(
            f"Brazilian corpus path: {self.brazilian_corpus_path}")
        self.logger.info(f"Machado path: {self.machado_path}")

        # Validate directory structure with more detailed logging
        if not self.brazilian_corpus_path.exists():
            self.logger.error(
                f"Brazilian corpus directory not found at {self.brazilian_corpus_path}")
            raise FileNotFoundError(
                f"Brazilian corpus directory not found at {self.brazilian_corpus_path}")
        if not self.machado_path.exists():
            self.logger.error(
                f"Machado de Assis directory not found at {self.machado_path}")
            raise FileNotFoundError(
                f"Machado de Assis directory not found at {self.machado_path}")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text using NLTK with Portuguese-specific processing."""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize and join back together
        words = word_tokenize(text, language='portuguese')
        return ' '.join(words)

    def chunk_text(self, text: str, overlap: int = 20) -> List[str]:
        """Split text into chunks with overlapping sentences for better context."""
        # Split into sentences first
        sentences = sent_tokenize(text, language='portuguese')
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            # Count words in the sentence
            words = sentence.split()
            sentence_length = len(words)

            # If a single sentence is longer than chunk_size, split it
            if sentence_length > self.chunk_size:
                # First, add any existing chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0

                # Split the long sentence into smaller pieces
                for i in range(0, len(words), self.chunk_size):
                    chunk_words = words[i:i + self.chunk_size]
                    chunks.append(' '.join(chunk_words))
                continue

            # If adding this sentence would exceed chunk size
            if current_word_count + sentence_length > self.chunk_size:
                # Save current chunk
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_word_count += sentence_length

        # Add any remaining text
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_file(self, file_path: Path, is_machado: int, author: str) -> List[Dict[str, Any]]:
        """Process a single text file and return chunks with metadata."""
        chunks = []
        # List of encodings to try
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        # Standardize author field - if not Machado, set to "Others"
        author = author if author == "Machado de Assis" else "Others"

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                # Preprocess text
                processed_text = self.preprocess_text(content)

                # Split into chunks
                text_chunks = self.chunk_text(processed_text)

                # Create entries for each chunk
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        'text': chunk,
                        'author': author,
                        'is_machado': is_machado,
                        'source_file': str(file_path.relative_to(self.base_path)),
                        'chunk_id': i
                    })

                self.logger.info(
                    f"Successfully processed {len(text_chunks)} chunks from: {file_path.name} using {encoding}")
                break  # If successful, break the encoding loop

            except UnicodeDecodeError:
                # If this was the last encoding to try
                if encoding == encodings[-1]:
                    self.logger.error(
                        f"Failed to decode {file_path} with any encoding")
                continue
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                break

        return chunks

    def process_brazilian_corpus(self) -> List[Dict[str, Any]]:
        """Process the Brazilian Portuguese corpus."""
        all_chunks = []

        self.logger.info(
            f"Searching for .txt files in: {self.brazilian_corpus_path}")

        try:
            # Modify the file search to include both top-level and subdirectory .txt files
            files = []
            # Add files from root directory
            files.extend(self.brazilian_corpus_path.glob("*.txt"))
            # Add files from author subdirectories
            for author_dir in self.brazilian_corpus_path.iterdir():
                if author_dir.is_dir():
                    files.extend(author_dir.glob("*.txt"))

            self.logger.info(
                f"Found {len(files)} .txt files in Brazilian corpus")

            if not files:
                self.logger.warning(
                    f"No .txt files found in {self.brazilian_corpus_path}")
                return all_chunks

            for file_path in files:
                # Skip guide file if it exists
                if "guide" in file_path.name.lower():
                    continue

                # Get author from parent directory name or "Unknown" if in root
                if file_path.parent.name == "Brazilian_Portugese_Corpus":
                    author = "Unknown"
                else:
                    author = file_path.parent.name

                self.logger.info(
                    f"Processing file: {file_path} with author: {author}")
                chunks = self.process_file(
                    file_path, is_machado=0, author=author)
                all_chunks.extend(chunks)

            self.logger.info(
                f"Total Brazilian corpus chunks processed: {len(all_chunks)}")
        except Exception as e:
            self.logger.error(f"Error processing Brazilian corpus: {str(e)}")

        return all_chunks

    def process_machado_corpus(self) -> List[Dict[str, Any]]:
        """Process Machado de Assis corpus."""
        all_chunks = []

        for file_path in self.machado_path.rglob("*.txt"):
            chunks = self.process_file(
                file_path, is_machado=1, author="Machado de Assis")
            all_chunks.extend(chunks)

        return all_chunks

    def create_unified_dataset(self) -> pd.DataFrame:
        """Create a unified dataset from both corpora."""
        # Process both corpora
        brazilian_chunks = self.process_brazilian_corpus()
        machado_chunks = self.process_machado_corpus()

        # Combine all chunks
        all_chunks = brazilian_chunks + machado_chunks

        # Create DataFrame
        df = pd.DataFrame(all_chunks)

        # Reorder columns
        df = df[['text', 'author', 'is_machado', 'source_file', 'chunk_id']]

        # print is_machado distribution
        print(f"is_machado distribution: {df['is_machado'].value_counts()}")

        return df


def main():
    # Debug directory structure
    base_dir = Path("data")
    print("Directory contents:")
    for path in base_dir.rglob("*"):
        print(f"  {path}")

    # Initialize processor with chunk size of 256 words
    processor = CorpusProcessor("data", chunk_size=256)

    # Create unified dataset
    dataset_df = processor.create_unified_dataset()

    # Save to CSV
    dataset_df.to_csv("data/processed/unified_corpus_chunks.csv", index=False)
    logging.info(
        f"Dataset saved to unified_corpus_chunks.csv with {len(dataset_df)} chunks")

    # Print some statistics
    logging.info(f"Total chunks: {len(dataset_df)}")
    logging.info(
        f"Machado chunks: {len(dataset_df[dataset_df['is_machado'] == 1])}")
    logging.info(
        f"Non-Machado chunks: {len(dataset_df[dataset_df['is_machado'] == 0])}")
    logging.info("\nAuthors distribution:")
    logging.info(dataset_df['author'].value_counts())
    logging.info("\nAverage words per chunk:")
    logging.info(dataset_df['text'].str.split().str.len().mean())


if __name__ == "__main__":
    main()
