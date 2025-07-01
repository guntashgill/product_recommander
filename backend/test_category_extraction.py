import json
import gzip
import re
from collections import Counter

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

def test_category_extraction(sample_size=1000):
    """Test category extraction with advanced logic"""
    print("Testing category extraction with advanced categorization...")
    categories = Counter()
    sample_count = 0
    electronics_count = 0
    
    with gzip.open('data/meta_Electronics.json.gz', 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
                
            try:
                product = json.loads(line)
                category = extract_category(product)
                categories[category] += 1
                sample_count += 1
                
                if category != 'Non-Electronics':
                    electronics_count += 1
                
                # Print first 10 for verification
                if i < 10:
                    title = product.get('title', '')[:70]
                    print(f"Product {i+1}: {title}... → {category}")
            except Exception as e:
                print(f"Error processing product: {str(e)}")
                categories['Error'] += 1
    
    print(f"\nProcessed {sample_count} products")
    
    # Avoid division by zero
    if sample_count > 0:
        print(f"Electronics products: {electronics_count} ({electronics_count/sample_count:.1%})")
    else:
        print("No products processed!")
        return False
    
    # Create counter for electronics categories only
    electronics_categories = Counter()
    for cat, count in categories.items():
        if cat not in ['Non-Electronics', 'Error']:
            electronics_categories[cat] = count
    
    print("\nCategory distribution for electronics:")
    for cat, count in electronics_categories.most_common():
        print(f"{cat}: {count} products")
    
    # Calculate success rate
    general_count = electronics_categories.get('General Electronics', 0)
    if electronics_count > 0:
        success_rate = 100 * (1 - general_count / electronics_count)
        print(f"\nSuccess rate: {success_rate:.1f}% of electronics products categorized")
    else:
        success_rate = 0
        print("\nNo electronics products found!")
    
    if success_rate > 70:
        print("✅ Category extraction is working well!")
        return True
    else:
        print("❌ Category extraction needs improvement")
        return False

if __name__ == '__main__':
    test_category_extraction()