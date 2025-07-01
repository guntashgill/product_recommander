from data_loader import load_amazon_data

if __name__ == '__main__':
    customers, products, interactions = load_amazon_data()
    print("Amazon data loaded successfully!")
    print(f"Customers: {len(customers)} records")
    print(f"Products: {len(products)} records")
    print(f"Interactions: {len(interactions)} records")