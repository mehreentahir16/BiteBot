"""
Database helper functions for the BiteBot agent.

Provides functions to query the restaurant database.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Database path
DB_PATH = Path(__file__).parent.parent / 'data' / 'restaurants.db'

def get_connection():
    """Get a connection to the database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Please run 'python scripts/prepare_data.py' first."
        )
    return sqlite3.connect(str(DB_PATH))

def dict_factory(cursor, row):
    """Convert database rows to dictionaries."""
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}

def search_restaurants(
    cuisine: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    min_stars: Optional[float] = None,
    max_price: Optional[int] = None,
    has_takeout: Optional[bool] = None,
    has_delivery: Optional[bool] = None,
    outdoor_seating: Optional[bool] = None,
    wheelchair_accessible: Optional[bool] = None,
    good_for_kids: Optional[bool] = None,
    has_wifi: Optional[bool] = None,
    accepts_reservations: Optional[bool] = None, 
    good_for_groups: Optional[bool] = None,  
    limit: int = 10
) -> List[Dict]:
    """
    Search for restaurants matching the given criteria.
    
    Args:
        cuisine: Cuisine type (e.g., 'Italian', 'Mexican', 'Chinese')
        city: City name (e.g., 'Philadelphia', 'Tampa')
        state: State abbreviation (e.g., 'PA', 'FL')
        min_stars: Minimum star rating (0-5)
        max_price: Maximum price range (1-4)
        accepts_reservations: Filter for restaurants that accept reservations  
        good_for_groups: Filter for restaurants good for groups 
        limit: Maximum number of results to return
        
    Returns:
        List of restaurant dictionaries
    """
    conn = get_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    
    query = "SELECT * FROM restaurants WHERE 1=1"
    params = []
    
   # Attribute filters (basic SQL filtering)
    if cuisine:
        query += " AND categories LIKE ?"
        params.append(f"%{cuisine}%")
    
    if city:
        query += " AND LOWER(city) = LOWER(?)"
        params.append(city)
    
    if state:
        query += " AND UPPER(state) = UPPER(?)"
        params.append(state)
    
    if min_stars is not None:
        query += " AND stars >= ?"
        params.append(min_stars)
    
    if max_price is not None:
        # Create the correct number of LIKE clauses based on max_price
        price_conditions = " OR ".join(["attributes LIKE ?"] * int(max_price))
        query += f" AND (attributes IS NULL OR {price_conditions})"
        
        # Add matching number of parameters
        for i in range(1, int(max_price) + 1):
            params.append(f'%"RestaurantsPriceRange2": "{i}"%')
    
    if has_takeout is not None:
        query += " AND attributes LIKE ?"
        params.append('%"RestaurantsTakeOut": "True"%' if has_takeout else '%"RestaurantsTakeOut": "False"%')
    
    if has_delivery is not None:
        query += " AND attributes LIKE ?"
        params.append('%"RestaurantsDelivery": "True"%' if has_delivery else '%"RestaurantsDelivery": "False"%')
    
    if outdoor_seating is not None:
        query += " AND attributes LIKE ?"
        params.append('%"OutdoorSeating": "True"%' if outdoor_seating else '%"OutdoorSeating": "False"%')
    
    if wheelchair_accessible is not None:
        query += " AND attributes LIKE ?"
        params.append('%"WheelchairAccessible": "True"%' if wheelchair_accessible else '%"WheelchairAccessible": "False"%')
    
    if good_for_kids is not None:
        query += " AND attributes LIKE ?"
        params.append('%"GoodForKids": "True"%' if good_for_kids else '%"GoodForKids": "False"%')
    
    if has_wifi is not None:
        query += " AND attributes LIKE ?"
        params.append('%"WiFi":%' if has_wifi else '')
    
    if accepts_reservations is not None:
        query += " AND attributes LIKE ?"
        params.append(f'%"RestaurantsReservations": "True"%' if accepts_reservations else '%"RestaurantsReservations": "False"%')
    
    if good_for_groups is not None:
        query += " AND attributes LIKE ?"
        params.append(f'%"RestaurantsGoodForGroups": "True"%' if good_for_groups else '%"RestaurantsGoodForGroups": "False"%')
    
    # Order by rating and review count
    query += " ORDER BY stars DESC, review_count DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    # Parse JSON fields and post-process for accurate filtering
    for result in results:
        if result.get('attributes'):
            try:
                result['attributes'] = json.loads(result['attributes'])
            except:
                result['attributes'] = {}
        else:
            result['attributes'] = {}
            
        if result.get('hours'):
            try:
                result['hours'] = json.loads(result['hours'])
            except:
                result['hours'] = {}
        else:
            result['hours'] = {}
    
    return results

def get_restaurant_by_id(business_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific restaurant.
    
    Args:
        business_id: The Yelp business ID
        
    Returns:
        Restaurant dictionary or None if not found
    """
    conn = get_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM restaurants WHERE business_id = ?", (business_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # Parse JSON fields
        if result.get('attributes'):
            try:
                result['attributes'] = json.loads(result['attributes'])
            except:
                result['attributes'] = {}
        else:
            result['attributes'] = {}
            
        if result.get('hours'):
            try:
                result['hours'] = json.loads(result['hours'])
            except:
                result['hours'] = {}
        else:
            result['hours'] = {}
    
    return result

def get_restaurant_by_name(name: str, city: Optional[str] = None) -> Optional[Dict]:
    """
    Get a restaurant by name (and optionally city for disambiguation).
    
    Args:
        name: Restaurant name
        city: City name (optional, for disambiguation)
        
    Returns:
        Restaurant dictionary or None if not found
    """
    conn = get_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    
    if city:
        cursor.execute(
            "SELECT * FROM restaurants WHERE LOWER(name) LIKE LOWER(?) AND LOWER(city) = LOWER(?) LIMIT 1",
            (f"%{name}%", city)
        )
    else:
        cursor.execute(
            "SELECT * FROM restaurants WHERE LOWER(name) LIKE LOWER(?) LIMIT 1",
            (f"%{name}%",)
        )
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # Parse JSON fields
        if result.get('attributes'):
            try:
                result['attributes'] = json.loads(result['attributes'])
            except:
                result['attributes'] = {}
        else:
            result['attributes'] = {}
            
        if result.get('hours'):
            try:
                result['hours'] = json.loads(result['hours'])
            except:
                result['hours'] = {}
        else:
            result['hours'] = {}
    
    return result

def is_open_now(hours: Optional[Dict]) -> tuple[bool, str]:
    """
    Check if a restaurant is currently open based on hours.
    
    Args:
        hours: Dictionary of hours (e.g., {"Monday": "8:0-22:0", ...})
        
    Returns:
        Tuple of (is_open: bool, message: str)
    """
    if not hours:
        return False, "Hours not available"
    
    now = datetime.now()
    day_name = now.strftime("%A")
    current_time = now.hour * 60 + now.minute  # Minutes since midnight
    
    day_hours = hours.get(day_name)
    if not day_hours:
        return False, f"No hours listed for {day_name}"
    
    # Parse hours (format: "8:0-22:0" or "0:0-0:0" for closed)
    try:
        open_time, close_time = day_hours.split('-')
        open_hour, open_min = map(int, open_time.split(':'))
        close_hour, close_min = map(int, close_time.split(':'))
        
        open_minutes = open_hour * 60 + open_min
        close_minutes = close_hour * 60 + close_min
        
        # Check if "0:0-0:0" (closed all day)
        if open_minutes == 0 and close_minutes == 0:
            return False, "Closed today"
        
        # Format open and close times
        open_str = f"{open_hour}:{open_min:02d}"
        close_str = f"{close_hour}:{close_min:02d}"
        
        # Check if current time is within operating hours
        if close_minutes < open_minutes:  # Crosses midnight
            is_open = current_time >= open_minutes or current_time < close_minutes
        else:
            is_open = open_minutes <= current_time < close_minutes
        
        if is_open:
            return True, f"Open now (closes at {close_str})"
        else:
            return False, f"Closed (opens at {open_str})"
            
    except Exception as e:
        return False, f"Unable to parse hours: {day_hours}"

def get_all_cities() -> List[str]:
    """Get a list of all cities in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT city FROM restaurants ORDER BY city")
    cities = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cities

def get_all_states() -> List[str]:
    """Get a list of all states in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT state FROM restaurants ORDER BY state")
    states = [row[0] for row in cursor.fetchall()]
    conn.close()
    return states