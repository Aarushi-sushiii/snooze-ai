import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def combine_text(row):
    parts = []
    for col in ['product_name', 'description', 'gender', 'occasion', 'color']:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    return ' '.join(parts)

def main():
    df = pd.read_csv('cleaned_dataset.csv')
    df['search_text'] = df.apply(combine_text, axis=1)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    product_embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=True)
    with open('product_embeddings.pkl', 'wb') as f:
        pickle.dump(product_embeddings, f)
    print('Embeddings generated and saved to product_embeddings.pkl')

if __name__ == '__main__':
    main()
