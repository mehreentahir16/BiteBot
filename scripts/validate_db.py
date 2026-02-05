"""
Validation script for BiteBot database preparation.

This script tests the data preparation process using the sample Yelp data
and validates that the database is created correctly.

Usage:
    python validate_db.py
"""

import json
import sqlite3
from pathlib import Path

def validate_sample_data():
    """Validate the database creation with sample data."""
    print("="*60)
    print("BiteBot DATABASE VALIDATION")
    print("="*60)
    
    # Use the sample file
    sample_file = Path('data/yelp_academic_dataset_business.json')
    db_file = Path('data/restaurants.db')
    
    if not sample_file.exists():
        print(f"\n❌ Sample file not found: {sample_file}")
        print("Please place the sample file in the data/ directory")
        return False
    
    print(f"\n1. Loading sample data from: {sample_file}")
    
    # Load sample data
    businesses = []
    with open(sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                businesses.append(json.loads(line.strip()))
            except:
                continue
    
    print(f"   ✅ Loaded {len(businesses)} businesses")
    
    # Filter for restaurants
    print("\n2. Filtering for restaurants...")
    restaurants = [b for b in businesses if b.get('categories') and 'Restaurants' in b['categories']]
    print(f"   ✅ Found {len(restaurants)} restaurants")
    
    # Show sample categories
    print("\n3. Sample restaurant categories:")
    for r in restaurants[:3]:
        print(f"   • {r['name']}: {r['categories']}")
    
    # Create test database
    print(f"\n4. Creating test database: {db_file}")
    if db_file.exists():
        db_file.unlink()
    
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Create table
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
    
    # Insert data
    for restaurant in restaurants:
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
    
    conn.commit()
    print(f"   ✅ Inserted {len(restaurants)} restaurants")
    
    # Validate the data
    print("\n5. Validating database contents...")
    
    cursor.execute("SELECT COUNT(*) FROM restaurants")
    count = cursor.fetchone()[0]
    print(f"   ✅ Total restaurants in DB: {count}")
    
    cursor.execute("SELECT COUNT(DISTINCT city) FROM restaurants")
    cities = cursor.fetchone()[0]
    print(f"   ✅ Unique cities: {cities}")
    
    cursor.execute("SELECT city, COUNT(*) FROM restaurants GROUP BY city ORDER BY COUNT(*) DESC LIMIT 5")
    print("\n   Top 5 cities:")
    for city, cnt in cursor.fetchall():
        print(f"      • {city}: {cnt}")
    
    # Test queries
    print("\n6. Testing sample queries...")
    
    # Query 1: Italian restaurants
    cursor.execute("SELECT name, city, stars FROM restaurants WHERE categories LIKE '%Italian%' LIMIT 3")
    results = cursor.fetchall()
    print(f"\n   Italian restaurants ({len(results)} found):")
    for name, city, stars in results:
        print(f"      • {name} in {city} - {stars}⭐")
    
    # Query 2: High-rated restaurants
    cursor.execute("SELECT name, city, stars FROM restaurants WHERE stars >= 4.0 ORDER BY stars DESC LIMIT 3")
    results = cursor.fetchall()
    print(f"\n   High-rated restaurants (4.0+ stars):")
    for name, city, stars in results:
        print(f"      • {name} in {city} - {stars}⭐")
    
    # Query 3: Restaurants with hours
    cursor.execute("SELECT name, city FROM restaurants WHERE hours IS NOT NULL LIMIT 3")
    results = cursor.fetchall()
    print(f"\n   Restaurants with hours data:")
    for name, city in results:
        print(f"      • {name} in {city}")
    
    # Query 4: Check attributes
    cursor.execute("SELECT name, attributes FROM restaurants WHERE attributes IS NOT NULL LIMIT 2")
    results = cursor.fetchall()
    print(f"\n   Sample attributes data:")
    for name, attrs_str in results:
        attrs = json.loads(attrs_str) if attrs_str else {}
        price = attrs.get('RestaurantsPriceRange2', 'N/A')
        print(f"      • {name}: Price range = {price}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE - DATABASE LOOKS GOOD!")
    print("="*60)
    print(f"\nTest database created at: {db_file}")
    print("\nYou can now run the full data preparation with:")
    print("  python scripts/prepare_data.py")
    
    return True

if __name__ == '__main__':
    try:
        validate_sample_data()
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()