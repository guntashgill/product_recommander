from flask import Flask, jsonify, request, make_response, send_from_directory
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import random
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

cluster_image_base64 = None

# Load data
customers = pd.read_csv('customers.csv')
products = pd.read_csv('products.csv')
interactions = pd.read_csv('interactions.csv')

# Segment customers
customers = customers.copy()
conditions = [
    (customers['age'] < 25),
    (customers['age'].between(25, 40)),
    (customers['age'] > 40)
]
segments = ['Young', 'Adult', 'Senior']
customers['age_segment'] = np.select(conditions, segments, default='Unknown')

customers['segment'] = (
    customers['age_segment'] + '_' + 
    customers['interests'] + '_' + 
    customers['location']
)

AB_TEST_GROUPS = {
    'A': 'content_based',
    'B': 'collaborative',
    'C': 'hybrid'
}

def log_ab_test(customer_id, group, recommendations):
    """Log A/B test results"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'customer_id': customer_id,
        'group': group,
        'recommendations': [r['product_id'] for r in recommendations]
    }
    
    with open('ab_test_logs.json', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def clamp_value(n, min_val=1, max_val=100):
    try:
        n = int(n)
        return max(min_val, min(n, max_val))
    except:
        return min_val

def get_popular_products(segment=None, n=5):
    """Get popular products based on interactions"""
    merged = interactions.merge(products, on='product_id')
    
    if segment:
        segment_customers = customers[
            customers['segment'].str.lower() == segment.lower()
        ]
        if not segment_customers.empty:
            merged = merged[merged['customer_id'].isin(segment_customers['customer_id'])]
    
    # Use interaction weights
    weights = {'view': 1, 'cart': 3, 'purchase': 5}
    merged['weight'] = merged['interaction_type'].map(weights)
    
    if not merged.empty:
        popular = merged.groupby(['product_id', 'title', 'brand', 'price']).agg(
            score=('weight', 'sum'),
            interactions=('weight', 'count')
        ).reset_index().sort_values('score', ascending=False).head(n)
        return popular.to_dict('records')
    
    return []

def get_content_based_recommendations(customer_id, n=5):
    """Content-based recommendations using product categories"""
    try:
        customer = customers[customers['customer_id'] == customer_id].iloc[0]
    except IndexError:
        return []

    # Split interests if they're comma-separated
    customer_interests = [interest.strip() for interest in customer['interests'].split(',')]

    # Get all unique categories from product catalog
    all_categories = sorted(set(
        cat.strip()
        for sublist in products['category'].dropna().str.split(',')
        for cat in sublist
    ))

    # Create customer feature vector
    customer_features = np.zeros(len(all_categories))
    for interest in customer_interests:
        if interest in all_categories:
            idx = all_categories.index(interest)
            customer_features[idx] = 1

    # Create product feature matrix
    product_features = []
    for _, row in products.iterrows():
        features = np.zeros(len(all_categories))
        if isinstance(row['category'], str):
            product_cats = [cat.strip() for cat in row['category'].split(',')]
            for cat in product_cats:
                if cat in all_categories:
                    idx = all_categories.index(cat)
                    features[idx] = 1
        product_features.append(features)

    # Compute cosine similarity
    similarity = cosine_similarity([customer_features], product_features)[0]
    products_rec = products.copy()
    products_rec['similarity'] = similarity

    # Filter out purchased items
    purchased = interactions[
        (interactions['customer_id'] == customer_id) &
        (interactions['interaction_type'] == 'purchase')
    ]['product_id'].values

    recs = products_rec[~products_rec['product_id'].isin(purchased)]
    recs = recs.sort_values('similarity', ascending=False).head(n)

    return recs[['product_id', 'title', 'brand', 'price', 'category']].to_dict('records')


def get_collaborative_filtering_recommendations(customer_id, n=5):
    """Get recommendations using collaborative filtering (user-based)"""
    # Create user-item matrix
    interactions['weight'] = interactions['interaction_type'].map(
        {'view': 1, 'cart': 3, 'purchase': 5}
    )
    
    user_item_matrix = interactions.pivot_table(
        index='customer_id',
        columns='product_id',
        values='weight',
        fill_value=0
    )
    
    # Convert to sparse matrix
    sparse_matrix = csr_matrix(user_item_matrix.values)
    
    # Create KNN model
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model.fit(sparse_matrix)
    
    # Find similar users
    if customer_id not in user_item_matrix.index:
        return []  # No interactions for this user
    
    user_idx = user_item_matrix.index.get_loc(customer_id)
    distances, indices = model.kneighbors(sparse_matrix[user_idx])
    
    # Get similar users (excluding the user themselves)
    similar_users = user_item_matrix.iloc[indices.flatten()[1:]].index
    
    # Get products from similar users
    similar_interactions = interactions[
        interactions['customer_id'].isin(similar_users)
    ]
    
    # Filter out products the customer has already purchased
    user_products = interactions[
        (interactions['customer_id'] == customer_id) & 
        (interactions['interaction_type'] == 'purchase')
    ]['product_id'].unique()
    
    recommendations = similar_interactions[
        (~similar_interactions['product_id'].isin(user_products))
    ]
    
    # Calculate recommendation scores
    if recommendations.empty:
        return []
    
    recommendations = recommendations.groupby('product_id').agg(
        score=('weight', 'sum')
    ).reset_index().sort_values('score', ascending=False).head(n)
    
    # Merge with product details
    recommendations = recommendations.merge(
        products, on='product_id', how='left'
    )
    
    return recommendations.to_dict('records')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Recommendation API is running"})

@app.route('/api/segments', methods=['GET'])
def get_segments():
    """Get available customer segments"""
    segments = customers['segment'].unique().tolist()
    return jsonify({"segments": segments})

@app.route('/api/popular', methods=['GET'])
def get_popular():
    """Get popular products for a segment"""
    segment = request.args.get('segment')
    n = clamp_value(request.args.get('n', 5))
    popular_products = get_popular_products(segment, n)
    return jsonify({
        "segment": segment,
        "recommendations": popular_products
    })

@app.route('/api/recommend/<string:customer_id>', methods=['GET'])
def recommend(customer_id):
    """Get personalized recommendations for a customer"""
    if customer_id not in customers['customer_id'].values:
        return jsonify({"error": "Customer not found"}), 404
    
    n = clamp_value(request.args.get('n', 5))
    
    # Randomly assign to A/B test group
    group = random.choice(list(AB_TEST_GROUPS.keys()))
    strategy = AB_TEST_GROUPS[group]
    
    # Get recommendations based on strategy
    if strategy == 'content_based':
        recommendations = get_content_based_recommendations(customer_id, n)
    elif strategy == 'collaborative':
        recommendations = get_collaborative_filtering_recommendations(customer_id, n)
    else:  # hybrid
        content_recs = get_content_based_recommendations(customer_id, n)
        collab_recs = get_collaborative_filtering_recommendations(customer_id, n)
        
        # Combine and deduplicate
        seen = set()
        recommendations = []
        for rec in content_recs + collab_recs:
            if rec['product_id'] not in seen:
                seen.add(rec['product_id'])
                recommendations.append(rec)
            if len(recommendations) >= n:
                break
    
    # Log A/B test
    log_ab_test(customer_id, group, recommendations)
    
    return jsonify({
        "customer_id": customer_id,
        "strategy": strategy,
        "recommendations": recommendations
    })

@app.route('/api/cluster_visualization', methods=['GET'])
def cluster_visualization():
    """Visualize customer clusters"""
    global cluster_image_base64
    
    # Generate image only once
    if cluster_image_base64 is None:
        # Prepare features
        features = customers[['age']].copy()
        
        # Encode categorical features
        for col in ['gender', 'interests']:
            le = LabelEncoder()
            features[col] = le.fit_transform(customers[col])
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Cluster customers
        kmeans = KMeans(n_clusters=5, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Reduce to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(scaled_features)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=cluster_labels,
            cmap='viridis',
            alpha=0.6
        )
        plt.title('Customer Segmentation')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Cluster')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        cluster_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
    
    return jsonify({"image": f"data:image/png;base64,{cluster_image_base64}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)