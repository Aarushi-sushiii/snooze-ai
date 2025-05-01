from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from image_search import get_similar_images_api
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Text Search Model & Embedding Cache ---
PRODUCT_EMBEDDINGS_PATH = 'product_embeddings.pkl'
PRODUCT_DF_PATH = 'cleaned_dataset.csv'
text_model = None
product_embeddings = None
product_df = None

def get_text_search_model_and_data():
    global text_model, product_embeddings, product_df
    if text_model is None:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
    if product_embeddings is None or product_df is None:
        with open(PRODUCT_EMBEDDINGS_PATH, 'rb') as f:
            product_embeddings = pickle.load(f)
        product_df = pd.read_csv(PRODUCT_DF_PATH)
    return text_model, product_embeddings, product_df

@app.route('/text_search', methods=['POST'])
def text_search():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    text_model, product_embeddings, product_df = get_text_search_model_and_data()
    query_embedding = text_model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, product_embeddings).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    results = []
    for idx in top_indices:
        row = product_df.iloc[idx]
        results.append({
            'title': row.get('product_name', 'N/A'),
            'url': row.get('product_link', ''),
            'image_url': row.get('image_link', '')
        })
    return jsonify({'results': results})

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/main')
def main():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        results = get_similar_images_api(filepath, top_k=5)
        return jsonify({'results': results})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
