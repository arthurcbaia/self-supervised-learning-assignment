import pandas as pd

path = 'data/processed/unified_corpus_chunks.csv'
df = pd.read_csv(path)

# print first row
print(len(df.iloc[0]['text'].split()))
