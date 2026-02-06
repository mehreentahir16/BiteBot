"""
Data Preparation Script for BiteBot Agent

This script:
1. Loads the Yelp academic dataset (yelp_academic_dataset_business.json)
2. Filters for businesses that have "Restaurants" in their categories
3. Creates a SQLite database with the filtered restaurant data
4. Prints statistics about the processed data

"""

import os
import json
import random
import sqlite3
import traceback
from pathlib import Path
from collections import Counter

def load_yelp_data(filepath):
    """Load Yelp business data from JSONL file."""
    print(f"Loading Yelp business data from {filepath}...")
    businesses = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                business = json.loads(line.strip())
                businesses.append(business)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(businesses):,} businesses")
    return businesses

def filter_restaurants(businesses):
    """Filter businesses to only include open restaurants."""
    print("Filtering for restaurants...")
    restaurants = []
    
    for business in businesses:
        categories = business.get('categories', '')
        is_open = business.get('is_open')
        if categories and 'Restaurants' in categories and is_open == 1:
            restaurants.append(business)
    
    print(f"Found {len(restaurants):,} restaurants")
    return restaurants

def sample_restaurants(restaurants, sample_size=5000, seed=42):
    """Randomly sample a fixed number of restaurants."""
    random.seed(seed)
    if len(restaurants) <= sample_size:
        return restaurants
    return random.sample(restaurants, sample_size)


def create_database(restaurants, db_path):
    """Create SQLite database and populate with restaurant data."""
    print(f"Creating SQLite database at {db_path}...")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create restaurants table
    cursor.execute('''
        CREATE TABLE restaurants (
            business_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            postal_code TEXT,
            latitude REAL,
            longitude REAL,
            stars REAL,
            review_count INTEGER,
            is_open INTEGER,
            categories TEXT,
            attributes TEXT,
            hours TEXT
        )
    ''')
    
    # Create indexes for common queries
    cursor.execute('CREATE INDEX idx_city ON restaurants(city)')
    cursor.execute('CREATE INDEX idx_state ON restaurants(state)')
    cursor.execute('CREATE INDEX idx_stars ON restaurants(stars)')
    cursor.execute('CREATE INDEX idx_categories ON restaurants(categories)')
    cursor.execute('CREATE INDEX idx_is_open ON restaurants(is_open)')
    cursor.execute('CREATE INDEX idx_name ON restaurants(name)')
    
    # Insert restaurant data
    inserted_count = 0
    for restaurant in restaurants:
        try:
            cursor.execute('''
                INSERT INTO restaurants VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                restaurant.get('business_id'),
                restaurant.get('name'),
                restaurant.get('address'),
                restaurant.get('city'),
                restaurant.get('state'),
                restaurant.get('postal_code'),
                restaurant.get('latitude'),
                restaurant.get('longitude'),
                restaurant.get('stars'),
                restaurant.get('review_count'),
                restaurant.get('is_open'),
                restaurant.get('categories'),
                json.dumps(restaurant.get('attributes')) if restaurant.get('attributes') else None,
                json.dumps(restaurant.get('hours')) if restaurant.get('hours') else None
            ))
            inserted_count += 1
        except Exception as e:
            print(f"Warning: Failed to insert restaurant {restaurant.get('name')}: {e}")
    
    conn.commit()
    conn.close()
    print(f"Database created successfully!")
    print(f"Inserted {inserted_count:,} restaurants into database")

def print_statistics(restaurants):
    """Print statistics about the restaurant data."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal restaurants: {len(restaurants):,}")
    
    # City distribution
    cities = Counter([r['city'] for r in restaurants])
    print(f"\nTotal cities: {len(cities):,}")
    print("\nTop 10 cities by restaurant count:")
    for city, count in cities.most_common(10):
        print(f"  {city}: {count:,}")
    
    # State distribution
    states = Counter([r['state'] for r in restaurants])
    print(f"\nTop 5 states:")
    for state, count in states.most_common(5):
        print(f"  {state}: {count:,}")
    
    # Rating distribution
    ratings = [r['stars'] for r in restaurants if r.get('stars')]
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        print(f"\nAverage rating: {avg_rating:.2f} stars")
    
    # Open vs closed
    open_count = sum(1 for r in restaurants if r.get('is_open') == 1)
    closed_count = len(restaurants) - open_count
    print(f"\nCurrently open: {open_count:,} ({open_count/len(restaurants)*100:.1f}%)")
    print(f"Currently closed: {closed_count:,} ({closed_count/len(restaurants)*100:.1f}%)")
    
    # Cuisine types (sample from first 200 restaurants)
    print("\nSample cuisine types:")
    cuisine_samples = set()
    for r in restaurants[:200]:
        cats = r.get('categories', '')
        if cats:
            for cat in cats.split(', '):
                cat = cat.strip()
                if cat and cat not in ['Restaurants', 'Food']:
                    cuisine_samples.add(cat)
                    if len(cuisine_samples) >= 15:
                        break
        if len(cuisine_samples) >= 15:
            break
    
    for cuisine in sorted(list(cuisine_samples)[:10]):
        print(f"  - {cuisine}")
    
    print("\n" + "="*50)

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data/yelp_dataset'
    input_file = data_dir / 'yelp_academic_dataset_business.json'
    output_db = data_dir / 'restaurants.db'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("\nPlease:")
        print("1. Create the 'data/' directory if it doesn't exist")
        print("2. Place 'yelp_academic_dataset_business.json' in the 'data/yelp_dataset' directory")
        print(f"   Expected location: {input_file}")
        return
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Load and filter data
        businesses = load_yelp_data(input_file)
        restaurants = filter_restaurants(businesses)
        
        if not restaurants:
            print("\nERROR: No restaurants found in the dataset!")
            print("Please check that your Yelp dataset contains businesses with 'Restaurants' in categories")
            return
        
        # Randomly sample 5000 restaurants
        restaurants = sample_restaurants(restaurants, sample_size=5000)
        
        # Create database
        create_database(restaurants, output_db)
        
        # Print statistics
        print_statistics(restaurants)
        
        print("\n✅ Data preparation complete!")
        print(f"Database location: {output_db}")
        print("\nYou can now run the agent")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()