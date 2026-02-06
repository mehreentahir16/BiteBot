"""
Build Pinecone review index from Yelp dataset.

Filters reviews to only include restaurants in our SQLite database,
generates embeddings via OpenAI, and uploads to Pinecone.

Usage:
    python scripts/build_review_index.py

Environment variables required:
    OPENAI_API_KEY
    PINECONE_API_KEY

Requires:
    pip install pinecone openai
"""

import os
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configuration
REVIEW_FILE = "data/yelp_dataset/yelp_academic_dataset_review.json"
DB_PATH = "data/restaurants.db"
INDEX_NAME = "bitebot-reviews"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # Pinecone batch upsert size
MAX_TEXT_LENGTH = 1000  # Characters to keep in metadata for display

# Filtering configuration - adjust these to control index size
# MIN_STARS = 3.0              # Only keep 3+ star reviews (filters out negative noise)
MIN_USEFUL_VOTES = 1         # Only keep reviews with at least 1 useful vote (quality signal)
MIN_REVIEW_DATE = "2022-01-01"  # Only keep reviews from last ~4 years (recent = relevant)
MAX_REVIEWS_PER_RESTAURANT = 5  # Cap reviews per restaurant (prevents dominance by popular spots)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def get_restaurant_business_ids():
    """Load all business_ids from our restaurant database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT business_id FROM restaurants")
    business_ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    print(f"✓ Loaded {len(business_ids)} restaurant IDs from database")
    return business_ids


def create_or_get_index():
    """Create Pinecone index if it doesn't exist."""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dimensions
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✓ Index '{INDEX_NAME}' created")
    else:
        print(f"✓ Index '{INDEX_NAME}' already exists")
    
    return pc.Index(INDEX_NAME)


def generate_embedding(text):
    """Generate OpenAI embedding for text."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def stream_reviews(filepath, valid_business_ids):
    """Stream reviews from JSON lines file, applying quality filters."""
    restaurant_counts = {}  # Track reviews per restaurant
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            business_id = review['business_id']
            
            # Filter 1: Must be in our restaurant set
            if business_id not in valid_business_ids:
                continue
            
            # # Filter 2: Minimum star rating
            # if review['stars'] < MIN_STARS:
            #     continue
            
            # Filter 3: Must have at least 1 useful vote (quality signal)
            if review.get('useful', 0) < MIN_USEFUL_VOTES:
                continue
            
            # Filter 4: Recent reviews only
            if review['date'] < MIN_REVIEW_DATE:
                continue
            
            # Filter 5: Per-restaurant limit (keep most useful reviews)
            # Track count per restaurant
            if business_id not in restaurant_counts:
                restaurant_counts[business_id] = 0
            
            if restaurant_counts[business_id] >= MAX_REVIEWS_PER_RESTAURANT:
                continue
            
            restaurant_counts[business_id] += 1
            yield review


def build_index():
    """Main ETL pipeline."""
    print("\n" + "="*60)
    print("BiteBot Review Index Builder")
    print("="*60 + "\n")
    
    # Show filtering config
    print("Filtering criteria:")
    # print(f"  • Minimum stars: {MIN_STARS}+")
    # print(f"  • Minimum useful votes: {MIN_USEFUL_VOTES}+")
    print(f"  • Reviews from: {MIN_REVIEW_DATE} onwards")
    print(f"  • Max per restaurant: {MAX_REVIEWS_PER_RESTAURANT}")
    print()
    
    # Step 1: Get valid business IDs
    valid_business_ids = get_restaurant_business_ids()
    
    # Step 2: Create or connect to index
    index = create_or_get_index()
    
    # Step 3: Process and upload in batches
    # (Skip counting - with filters it's too slow on 4.7M reviews)
    print("\nProcessing reviews with filters applied...")
    print(f"(Batch size: {BATCH_SIZE}, progress updates every 100 reviews)\n")
    
    batch = []
    processed = 0
    skipped = 0
    
    for review in stream_reviews(REVIEW_FILE, valid_business_ids):
        try:
            # Generate embedding
            embedding = generate_embedding(review['text'])
            
            # Prepare vector
            vector = {
                "id": review['review_id'],
                "values": embedding,
                "metadata": {
                    "business_id": review['business_id'],
                    "stars": float(review['stars']),
                    "date": review['date'],
                    "useful": int(review.get('useful', 0)),
                    "funny": int(review.get('funny', 0)),
                    "cool": int(review.get('cool', 0)),
                    "text": review['text'][:MAX_TEXT_LENGTH]  # Truncate for storage
                }
            }
            
            batch.append(vector)
            processed += 1
            
            # Upload batch when full
            if len(batch) >= BATCH_SIZE:
                index.upsert(vectors=batch)
                batch = []
                print(f"  ✓ Processed {processed} reviews...", end='\r')
            
        except Exception as e:
            skipped += 1
            if skipped % 100 == 0:
                print(f"\n  ⚠ Skipped {skipped} reviews due to errors")
            continue
    
    # Upload remaining batch
    if batch:
        index.upsert(vectors=batch)
    
    print(f"\n\n✅ Index build complete!")
    print(f"   Processed: {processed} reviews")
    print(f"   Skipped: {skipped} reviews")
    print(f"   Restaurants: {valid_business_ids.__len__()}")
    print(f"\n   Index stats: {index.describe_index_stats()}")
    
    # Estimate costs
    avg_tokens_per_review = 200
    total_tokens = processed * avg_tokens_per_review
    cost_estimate = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
    print(f"\n   Estimated embedding cost: ${cost_estimate:.2f}")


if __name__ == "__main__":
    if not os.path.exists(REVIEW_FILE):
        print(f"❌ Error: {REVIEW_FILE} not found")
        print("   Download from: https://www.yelp.com/dataset")
        exit(1)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: {DB_PATH} not found")
        print("   Run the restaurant database setup script first")
        exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in .env")
        exit(1)
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY not set in .env")
        exit(1)
    
    build_index()
