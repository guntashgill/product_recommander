import pandas as pd
import gzip
import json
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
import re
from collections import defaultdict, Counter

# Comprehensive electronics keywords
ELECTRONICS_KEYWORDS = {
    'phone', 'camera', 'headphone', 'headset', 'earbud', 'earphone', 
    'speaker', 'laptop', 'tablet', 'tv', 'television', 'monitor', 
    'router', 'printer', 'keyboard', 'mouse', 'gaming', 'battery', 
    'charger', 'storage', 'software', 'player', 'usb', 'bluetooth', 
    'cable', 'adapter', 'hard drive', 'memory card', 'sd card', 'ssd', 
    'hdmi', 'display', 'screen', 'microphone', 'amplifier', 'console', 
    'video game', 'graphics card', 'motherboard', 'processor', 'ram', 
    'memory', 'nvr', 'dvr', 'surveillance', 'mp3', 'media', 'dvd', 
    'cd', 'led', 'lcd', 'oled', 'ssd', 'hdd', 'flash drive', 'pendrive',
    'cctv', 'ip camera', 'wireless', 'charger', 'power bank', 'adapter',
    'cable', 'hub', 'dock', 'stylus', 'smartwatch', 'fitbit', 'drone',
    'gps', 'projector', 'scanner', 'toner', 'ink', 'tripod', 'lens',
    'filter', 'bundle', 'kit', 'gadget', 'electronic', 'device', 'tech',
    'nvr', 'dvr', 'surveillance', 'recorder', 'security', 'alarm'
}

# Category mapping with priority
CATEGORY_MAPPING = [
    # Phones
    (['phone', 'smartphone', 'iphone', 'android', 'mobile'], 'Phones'),
    
    # Audio
    (['headphone', 'earbud', 'earphone', 'headset', 'speaker', 
      'audio', 'sound', 'mic', 'microphone'], 'Audio'),
      
    # Computing
    (['laptop', 'notebook', 'ultrabook', 'chromebook', 'desktop', 
      'computer', 'cpu', 'processor', 'ram', 'ssd', 'hdd'], 'Computing'),
      
    # Tablets
    (['tablet', 'ipad', 'kindle fire', 'android tablet'], 'Tablets'),
    
    # TVs
    (['tv', 'television', 'oled', 'led', 'lcd', 'display', 'monitor'], 'TVs'),
    
    # Cameras
    (['camera', 'dslr', 'mirrorless', 'lens', 'tripod', 'filter'], 'Cameras'),
    
    # Gaming
    (['gaming', 'gamer', 'console', 'playstation', 'xbox', 'nintendo', 
      'steam', 'controller', 'joystick'], 'Gaming'),
      
    # Accessories
    (['battery', 'charger', 'adapter', 'cable', 'hub', 'dock', 'case', 
      'cover', 'stand', 'mount', 'stylus', 'sleeve'], 'Accessories'),
      
    # Storage
    (['storage', 'hard drive', 'memory card', 'sd card', 'ssd', 
      'flash drive', 'pendrive', 'external', 'usb drive'], 'Storage'),
      
    # Networking
    (['router', 'modem', 'switch', 'access point', 'repeater', 
      'network', 'ethernet', 'wifi', 'wireless'], 'Networking'),
      
    # Printers
    (['printer', 'ink', 'toner', 'scanner', 'copier'], 'Printers'),
    
    # Software
    (['software', 'antivirus', 'os', 'operating system', 'microsoft office', 
      'adobe', 'license', 'subscription'], 'Software'),
    
    # Media Players
    (['player', 'media player', 'streaming', 'dvd', 'blu-ray', 'cd'], 'Media Players'),
    
    # Other Electronics
    (['drone', 'gps', 'projector', 'smart home', 'iot', 'wearable', 
      'smartwatch', 'fitbit', 'surveillance', 'cctv', 'nvr', 'dvr'], 'Other Electronics')
]

def is_electronics(title, description=""):
    """Determine if a product is electronics based on title/description"""
    text = (title + " " + description).lower()
    return any(keyword in text for keyword in ELECTRONICS_KEYWORDS)

def extract_category(product):
    """Advanced category extraction with title analysis"""
    title = product.get('title', '').lower()
    brand = product.get('brand', '').lower()
    
    # Check if it's actually electronics
    if not is_electronics(title):
        return 'Non-Electronics'
    
    # Try categories field if available
    categories = product.get('categories', [])
    if categories and isinstance(categories, list):
        # Flatten category trees
        all_cats = [cat.strip().lower() for tree in categories if isinstance(tree, list) for cat in tree]
        
        # Check for specific category matches
        for cat in all_cats:
            for keywords, category in CATEGORY_MAPPING:
                if any(kw in cat for kw in keywords):
                    return category
                    
    # Analyze title for category keywords
    for keywords, category in CATEGORY_MAPPING:
        if any(kw in title for kw in keywords):
            return category
    
    # Analyze brand for category hints
    brand_mapping = {
        'sony': 'Audio', 'bose': 'Audio', 'jbl': 'Audio',
        'samsung': 'Phones', 'apple': 'Phones', 'xiaomi': 'Phones',
        'logitech': 'Accessories', 'anker': 'Accessories',
        'nvidia': 'Gaming', 'razer': 'Gaming', 'steelseries': 'Gaming',
        'canon': 'Cameras', 'nikon': 'Cameras', 'dji': 'Other Electronics'
    }
    for brand_prefix, category in brand_mapping.items():
        if brand_prefix in brand:
            return category
    
    return 'General Electronics'

def parse_product(line):
    """Fast product parsing with direct category extraction"""
    try:
        product = json.loads(line)
        asin = product.get('asin', '')
        title = product.get('title', 'Unknown Product')
        brand = product.get('brand', 'Unknown Brand')
        
        # Extract price efficiently
        price = product.get('price', 0)
        if isinstance(price, str):
            try:
                price = float(price.replace('$', '').replace(',', '').strip())
            except:
                price = 0.0
        elif not isinstance(price, (int, float)):
            price = 0.0
            
        # Extract category using our advanced method
        category = extract_category(product)
        
        return {
            'product_id': asin,
            'title': title,
            'brand': brand,
            'price': price,
            'category': category
        }
    except Exception as e:
        print(f"Error parsing product: {str(e)[:100]}")
        return None

def parse_review(line):
    """Fast review parsing"""
    try:
        review = json.loads(line)
        return {
            'customer_id': review.get('reviewerID', ''),
            'product_id': review.get('asin', ''),
            'rating': review.get('overall', 0),
            'timestamp': review.get('unixReviewTime', 0)
        }
    except Exception as e:
        print(f"Error parsing review: {str(e)[:100]}")
        return None

def load_amazon_data():
    print("Loading product metadata...")
    start = time.time()
    
    # Parallel product processing
    products = []
    with gzip.open('data/meta_Electronics.json.gz', 'rt', encoding='utf-8') as f:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.imap(parse_product, f, chunksize=1000)
            for i, product in enumerate(results):
                if product:
                    products.append(product)
                # Print progress every 10,000 products
                if i % 10000 == 0:
                    print(f"Processed {i} products...")
    
    products_df = pd.DataFrame(products)
    print(f"Loaded {len(products_df)} products in {time.time()-start:.1f} seconds")
    
    # Diagnostic: Show category distribution
    print("\nProduct Category Distribution:")
    print(products_df['category'].value_counts().head(20))
    
    print("Loading reviews...")
    start = time.time()
    
    # Parallel review processing
    reviews = []
    with gzip.open('data/Electronics.json.gz', 'rt', encoding='utf-8') as f:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.imap(parse_review, f, chunksize=1000)
            for i, review in enumerate(results):
                if review:
                    reviews.append(review)
                # Print progress every 50,000 reviews
                if i % 50000 == 0:
                    print(f"Processed {i} reviews...")
    
    interactions_df = pd.DataFrame(reviews)
    print(f"Loaded {len(interactions_df)} interactions in {time.time()-start:.1f} seconds")
    
    # Map ratings to interactions
    interactions_df['interaction_type'] = interactions_df['rating'].apply(
        lambda r: 'view' if r < 3 else 'cart' if r == 3 else 'purchase'
    )
    
    # Convert timestamp
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], unit='s')
    
    # Optimized customer interest calculation
    print("Calculating customer interests...")
    start = time.time()
    
    # Step 1: Create a product_id -> category mapping dictionary
    product_category_map = {}
    for _, row in products_df.iterrows():
        product_category_map[row['product_id']] = row['category']
    
    # Step 2: Process interactions in chunks
    customer_interests = defaultdict(lambda: defaultdict(int))
    chunk_size = 1000000
    total_interactions = len(interactions_df)
    
    print(f"Processing {total_interactions} interactions in chunks...")
    
    for i in range(0, total_interactions, chunk_size):
        chunk = interactions_df.iloc[i:i+chunk_size]
        for _, row in chunk.iterrows():
            customer_id = row['customer_id']
            product_id = row['product_id']
            
            # Get category from map or default
            category = product_category_map.get(product_id, 'General Electronics')
            
            # Skip non-electronics products
            if category == 'Non-Electronics':
                continue
                
            # Update category count for customer
            customer_interests[customer_id][category] += 1
        
        # Print progress
        processed = min(i + chunk_size, total_interactions)
        print(f"Processed {processed}/{total_interactions} interactions ({processed/total_interactions:.1%})")
    
    # Step 3: Extract top interests
    interests_list = []
    for customer_id, categories in customer_interests.items():
        # Get top 2 categories
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:2]
        top_categories = [cat for cat, _ in sorted_cats]
        interests_list.append({
            'customer_id': customer_id,
            'interests': ','.join(top_categories) if top_categories else 'General Electronics'
        })
    
    # Convert to DataFrame
    customer_interests_df = pd.DataFrame(interests_list)
    
    # Create customer metadata
    customer_ids = interactions_df['customer_id'].unique()
    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': np.random.randint(18, 70, size=len(customer_ids)),
        'gender': np.random.choice(['M', 'F', 'Other'], len(customer_ids), p=[0.4, 0.55, 0.05]),
        'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], len(customer_ids))
    })
    
    # Merge interests
    customers_df = customers_df.merge(customer_interests_df, on='customer_id', how='left')
    customers_df['interests'] = customers_df['interests'].fillna('General Electronics')
    
    print(f"Processed {len(customers_df)} customers in {time.time()-start:.1f} seconds")
    
    # Save to CSV
    print("Saving data...")
    products_df.to_csv('products.csv', index=False)
    interactions_df[['customer_id', 'product_id', 'interaction_type', 'timestamp']].to_csv('interactions.csv', index=False)
    customers_df.to_csv('customers.csv', index=False)
    
    print("\nData generation complete!")
    print(f"Customers: {len(customers_df)}")
    print(f"Products: {len(products_df)}")
    print(f"Interactions: {len(interactions_df)}")
    print("\nTop 10 interests:")
    print(customers_df['interests'].value_counts().head(10))
    
    return customers_df, products_df, interactions_df
#t