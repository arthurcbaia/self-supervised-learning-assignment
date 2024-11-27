import os
from pathlib import Path

def remove_duplicate_files(directory):
    # Convert directory to Path object
    dir_path = Path(directory)
    
    # Keep track of seen filenames
    seen_files = set()
    
    # Iterate through all files in directory and subdirectories
    for file_path in dir_path.rglob('*.txt'):
        filename = file_path.name
        
        if filename in seen_files:
            # If we've seen this filename before, remove the duplicate
            print(f"Removing duplicate file: {file_path}")
            os.remove(file_path)
        else:
            # Add filename to our set of seen files
            seen_files.add(filename)
            print(f"Found new file: {file_path}")

# Example usage
directory_path = "data/Brazilian_Portugese_Corpus"
remove_duplicate_files(directory_path)