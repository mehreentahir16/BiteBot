"""
Agent tools for the BiteBot restaurant assistant.

These tools allow the LLM to interact with the restaurant database.
"""

import re
import json
import uuid
import logging
import threading
import dateparser
import contextvars
from typing import Optional
from datetime import datetime
from langchain.tools import tool

from src.database import (
    search_restaurants,
    get_restaurant_by_name,
    get_restaurant_by_id,
    is_open_now,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tool context – generic key/value store for inter-tool state.
# Persisted across HTTP requests via Flask session (see app.py / agent.py).

_tool_contexts: dict = {}           # thread_id -> {key: value}
_tool_ctx_lock = threading.Lock()   # guards _tool_contexts
_active_session: contextvars.ContextVar[str] = contextvars.ContextVar(
    '_active_session', default=None
)

def set_active_session(thread_id: str):
    """Bind the current execution context to a session. Called by run_agent before agent.invoke() so that tools land in the right bucket."""
    _active_session.set(thread_id)


def set_tool_context(key: str, value):
    """Store a value in the current session's tool context."""
    sid = _active_session.get()
    with _tool_ctx_lock:
        if sid not in _tool_contexts:
            _tool_contexts[sid] = {}
        _tool_contexts[sid][key] = value


def get_tool_context(key: str):
    """Retrieve a value from the current session's tool context."""
    sid = _active_session.get()
    if sid is None:
        return None
    with _tool_ctx_lock:
        return _tool_contexts.get(sid, {}).get(key)


def clear_tool_context(key: str):
    """Remove a key from the current session's tool context."""
    sid = _active_session.get()
    if sid is None:
        return
    with _tool_ctx_lock:
        if sid in _tool_contexts:
            _tool_contexts[sid].pop(key, None)

@tool
def search_restaurants_tool(
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
) -> str:
    """Search for restaurants based on criteria like cuisine, location, rating, price, and amenities.

    Args:
        cuisine: Cuisine type (e.g., 'Italian', 'Mexican', 'Chinese')
        city: City name (e.g., 'Philadelphia', 'Tampa')
        state: State abbreviation (e.g., 'PA', 'FL')
        min_stars: Minimum rating (1.0-5.0)
        max_price: Max price range (1-4)
        has_takeout: Takeout availability
        has_delivery: Delivery availability
        outdoor_seating: Outdoor seating
        wheelchair_accessible: Wheelchair accessibility
        good_for_kids: Kid-friendly
        has_wifi: WiFi availability
        accepts_reservations: Accepts reservations
        good_for_groups: Good for groups
        limit: Max results (default 10)
    """
    logger.info(f"Searching restaurants: cuisine={cuisine}, city={city}, state={state}")

    try:
        results = search_restaurants(
            cuisine=cuisine, city=city, state=state,
            min_stars=min_stars, max_price=max_price,
            has_takeout=has_takeout, has_delivery=has_delivery,
            outdoor_seating=outdoor_seating,
            wheelchair_accessible=wheelchair_accessible,
            good_for_kids=good_for_kids, has_wifi=has_wifi,
            accepts_reservations=accepts_reservations,
            good_for_groups=good_for_groups, limit=limit
        )

        logger.info(f"Found {len(results)} restaurants")

        if not results:
            return "No restaurants found matching your criteria. Try broadening your search."

        output = f"Found {len(results)} restaurant(s):\n\n"
        for i, restaurant in enumerate(results, 1):
            price_range = "N/A"
            if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
                price = restaurant['attributes'].get('RestaurantsPriceRange2')
                if price:
                    try:
                        price_range = '$' * int(price)
                    except (ValueError, TypeError):
                        pass

            output += f"{i}. **{restaurant['name']}**\n"
            output += f"   📍 {restaurant['address']}, {restaurant['city']}, {restaurant['state']}\n"
            output += f"   ⭐ {restaurant['stars']} stars ({restaurant['review_count']} reviews)\n"
            output += f"   💰 {price_range}\n"
            output += f"   🍽️  {restaurant['categories']}\n"
            output += f"   Status: {'🟢 Open' if restaurant['is_open'] == 1 else '🔴 Closed'}\n"
            output += f"   ID: {restaurant['business_id']}\n\n"

        return output

    except Exception as e:
        logger.error(f"Error searching restaurants: {e}", exc_info=True)
        return f"Error searching restaurants: {str(e)}"

@tool
def get_restaurant_details_tool(
    name: Optional[str] = None,
    city: Optional[str] = None,
    business_id: Optional[str] = None
) -> str:
    """Get detailed information about a specific restaurant.

    Args:
        name: Restaurant name
        city: City name (optional, for disambiguation)
        business_id: Yelp business ID (alternative to name)
    """
    logger.info(f"Getting details: name={name}, city={city}, business_id={business_id}")

    try:
        if business_id:
            restaurant = get_restaurant_by_id(business_id)
        elif name:
            restaurant = get_restaurant_by_name(name, city)
        else:
            return "Error: Please provide either 'name' or 'business_id'"

        if not restaurant:
            return "Restaurant not found. Please check the name and try again."

        logger.info(f"Found restaurant: {restaurant['name']}")

        output = f"**{restaurant['name']}**\n\n"
        output += f"📍 Address:\n   {restaurant['address']}\n   {restaurant['city']}, {restaurant['state']} {restaurant['postal_code']}\n\n"
        output += f"⭐ Rating: {restaurant['stars']} stars ({restaurant['review_count']} reviews)\n\n"

        if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
            price = restaurant['attributes'].get('RestaurantsPriceRange2')
            if price:
                try:
                    output += f"💰 Price Range: {'$' * int(price)}\n\n"
                except (ValueError, TypeError):
                    pass

        output += f"🍽️  Categories: {restaurant['categories']}\n\n"
        output += f"Status: {'🟢 Currently Open' if restaurant['is_open'] == 1 else '🔴 Currently Closed'}\n\n"

        if restaurant.get('hours') and isinstance(restaurant['hours'], dict):
            output += "🕒 Hours:\n"
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                output += f"   {day}: {restaurant['hours'].get(day, 'N/A')}\n"
            output += "\n"

        if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
            output += "ℹ️  Amenities & Features:\n"
            output += json.dumps(restaurant['attributes'], indent=2)
            output += "\n\n"

        output += f"🆔 Business ID: {restaurant['business_id']}\n"
        return output

    except Exception as e:
        logger.error(f"Error getting restaurant details: {e}", exc_info=True)
        return f"Error getting restaurant details: {str(e)}"

@tool
def check_availability_tool(
    name: Optional[str] = None,
    city: Optional[str] = None,
    business_id: Optional[str] = None,
    date: Optional[str] = None,
    time: Optional[str] = None,
    party_size: int = 2
) -> str:
    """Check if a restaurant is open now OR check table availability.

    Args:
        name: Restaurant name
        city: City name (optional)
        business_id: Yelp business ID (alternative to name)
        date: Date exactly as user said it - "today", "tomorrow", "this friday", "next thursday", "2026-02-15", etc.
              Do NOT convert to a date yourself — pass the user's words directly.
        time: Time in ANY format - "7pm", "19:00", "7:30 PM", etc.
        party_size: Number of people 
    """
    logger.info(f"Checking availability: name={name}, date={date}, time={time}, party_size={party_size}")

    try:
        if business_id:
            restaurant = get_restaurant_by_id(business_id)
        elif name:
            restaurant = get_restaurant_by_name(name, city)
        else:
            return "Error: Please provide either 'name' or 'business_id'"

        if not restaurant:
            return "Restaurant not found."

        logger.info(f"Found restaurant: {restaurant['name']}")

        accepts_reservations = False
        if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
            accepts_reservations = restaurant['attributes'].get('RestaurantsReservations') == 'True'

        now = datetime.now()

        if time or date:
            if not accepts_reservations:
                return f"❌ **{restaurant['name']}** does not accept reservations. This is a walk-in only restaurant."

            # parse date 
            if date:
                reservation_date = dateparser.parse(
                    date,
                    settings={
                        'RELATIVE_BASE': now,
                        'PREFER_DATES_FROM': 'future',
                        'STRICT_PARSING': False,
                        'RETURN_AS_TIMEZONE_AWARE': False,
                    },
                    languages=['en']
                )

                if not reservation_date:
                    return (f"Error: Could not understand date '{date}'. "
                            "Try 'today', 'tomorrow', 'this friday', 'next thursday', or 'YYYY-MM-DD'.")

            else:
                reservation_date = now

            # parse time 
            if time:
                time_str = f"{reservation_date.strftime('%Y-%m-%d')} {time}"
                parsed_dt = dateparser.parse(time_str, settings={'RETURN_AS_TIMEZONE_AWARE': False})

                if parsed_dt:
                    reservation_time = parsed_dt.strftime('%H:%M')
                else:
                    m = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', time.lower())
                    if m:
                        hour, minute = int(m.group(1)), int(m.group(2) or 0)
                        if m.group(3) == 'pm' and hour != 12:
                            hour += 12
                        elif m.group(3) == 'am' and hour == 12:
                            hour = 0
                        reservation_time = f"{hour:02d}:{minute:02d}"
                    else:
                        return f"Error: Could not understand time '{time}'. Try '7pm', '19:00', or '7:30 PM'."

            else:
                reservation_time = now.strftime('%H:%M')

            # check hours
            day_name = reservation_date.strftime("%A")

            if restaurant.get('hours') and day_name in restaurant['hours']:
                day_hours = restaurant['hours'][day_name]

                req_h, req_m  = reservation_time.split(':')
                req_minutes   = int(req_h) * 60 + int(req_m)
                open_time, close_time = day_hours.split('-')
                oh, om        = open_time.split(':')
                open_minutes  = int(oh) * 60 + int(om)
                ch, cm        = close_time.split(':')
                close_minutes = int(ch) * 60 + int(cm)

                if open_minutes == 0 and close_minutes == 0:
                    return f"❌ **{restaurant['name']}** is closed on {day_name}."

                if close_minutes < open_minutes:
                    time_ok = req_minutes >= open_minutes or req_minutes < close_minutes
                else:
                    time_ok = open_minutes <= req_minutes < close_minutes

                if not time_ok:
                    return (f"❌ **{restaurant['name']}** is not open at {reservation_time} on {day_name}.\n"
                            f"Hours: {day_hours}")

                # ✅ table available – stash in tool_context for make_reservation
                date_str = reservation_date.strftime("%Y-%m-%d")

                set_tool_context('availability', {
                    'date': date_str,
                    'time': reservation_time,
                    'party_size': party_size,
                    'restaurant': restaurant['name'],
                })
                logger.info("Availability confirmed and stored in context")

                output  = f"✅ **{restaurant['name']}**\n"
                output += f"📅 {date_str} ({day_name}) at {reservation_time}\n"
                output += f"👥 Party of {party_size}\n\n"
                output += f"Table available! Restaurant is open from {open_time} to {close_time}.\n"
                output += f"📍 {restaurant['address']}, {restaurant['city']}, {restaurant['state']}\n"
                return output
            else:
                return f"Hours not available for {day_name} at **{restaurant['name']}**."

        # no date/time → just check if open now 
        is_open, message = is_open_now(restaurant.get('hours'))
        output  = f"**{restaurant['name']}** in {restaurant['city']}, {restaurant['state']}\n"
        output += f"✅ {message}" if is_open else f"❌ {message}"

        if restaurant.get('hours'):
            today = now.strftime("%A")
            if (today_hours := restaurant['hours'].get(today)):
                output += f"\nToday's hours: {today_hours}"

        output += f"\n\n{'✓' if accepts_reservations else '✗'} This restaurant {'accepts' if accepts_reservations else 'does NOT accept'} reservations."
        return output

    except Exception as e:
        logger.error(f"Error checking availability: {e}", exc_info=True)
        return f"Error: {str(e)}"

@tool
def make_reservation_tool(
    customer_name: str,
    name: Optional[str] = None,
    city: Optional[str] = None,
    business_id: Optional[str] = None,
    party_size: int = 2,
    customer_phone: Optional[str] = None,
    special_requests: Optional[str] = None
) -> str:
    """Make a reservation at a restaurant.

    ONLY call this after:
    1. check_availability_tool has confirmed a table is available
    2. The user has explicitly said yes / confirmed
    3. The user has provided their real name

    Date and time are automatically pulled from the last availability check —
    do NOT pass them yourself.

    Args:
        customer_name: Customer's REAL name (e.g. "Sarah Johnson"). Never use placeholders.
        name: Restaurant name
        city: City (optional)
        business_id: Yelp business ID (optional)
        party_size: Number of guests
        customer_phone: Contact phone (optional)
        special_requests: Any special requests (optional)
    """
    logger.info(f"Making reservation for {customer_name} at restaurant: {name}")

    try:
        # validate customer name 
        if not customer_name or not customer_name.strip():
            return "❌ Please provide the name you'd like the reservation under."

        generic_names = ['guest', 'user', 'customer', 'reservation', 'table', 'name', 'person', 'client']
        if customer_name.lower().strip() in generic_names:
            return f"❌ '{customer_name}' is a placeholder. Please ask the user for their actual name."

        if len(customer_name.strip()) < 2:
            return f"❌ '{customer_name}' is too short. Please ask for the user's full name."

        # resolve restaurant
        if business_id:
            restaurant = get_restaurant_by_id(business_id)
        elif name:
            restaurant = get_restaurant_by_name(name, city)
        else:
            return "Error: Please provide a restaurant name."

        if not restaurant:
            return "Restaurant not found."

        # reservations supported? 
        accepts_reservations = False
        if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
            accepts_reservations = restaurant['attributes'].get('RestaurantsReservations') == 'True'

        if not accepts_reservations:
            return f"❌ **{restaurant['name']}** does not accept reservations (walk-in only)."

        # pull date/time from tool_context
        availability = get_tool_context('availability')
        if not availability:
            return "❌ Please check availability first before making a reservation."

        reservation_date_str = availability['date']
        reservation_time_str = availability['time']
        party_size            = availability['party_size']

        # sanity checks 
        reservation_date = datetime.strptime(reservation_date_str, '%Y-%m-%d')
        if reservation_date.date() < datetime.now().date():
            return "Error: The availability date is in the past. Please check availability again."

        # build confirmation 
        reservation_id = str(uuid.uuid4())[:8]

        reservation = {
            'reservation_id':   reservation_id,
            'restaurant_name':  restaurant['name'],
            'restaurant_id':    restaurant['business_id'],
            'address':          f"{restaurant['address']}, {restaurant['city']}, {restaurant['state']}",
            'date':             reservation_date_str,
            'time':             reservation_time_str,
            'party_size':       party_size,
            'customer_name':    customer_name,
            'customer_phone':   customer_phone or 'Not provided',
            'special_requests': special_requests or 'None',
            'status':           'confirmed',
            'created_at':       datetime.now().isoformat()
        }

        # Store reservation in tool context instead of embedding in string
        set_tool_context('reservation', reservation)
        logger.info(f"Reservation created: {reservation_id}")

        # Reservation confirmed — availability consumed, clear it
        clear_tool_context('availability')

        output  = f"🎉 **Reservation Confirmed!**\n\n"
        output += f"📋 Confirmation #: {reservation_id}\n\n"
        output += f"🍽️  **{restaurant['name']}**\n"
        output += f"📍 {reservation['address']}\n\n"
        output += f"📅 Date: {reservation_date_str}\n"
        output += f"🕐 Time: {reservation_time_str}\n"
        output += f"👥 Party Size: {party_size}\n"
        output += f"👤 Name: {customer_name}\n"
        output += f"📞 Phone: {reservation['customer_phone']}\n"

        if special_requests:
            output += f"💬 Special Requests: {special_requests}\n"

        output += f"\n✅ Your table is reserved!\n"

        return output

    except Exception as e:
        logger.error(f"Error making reservation: {e}", exc_info=True)
        return f"Error: {str(e)}"

restaurant_tools = [
    search_restaurants_tool,
    get_restaurant_details_tool,
    check_availability_tool,
    make_reservation_tool
]

# restaurant agent is the primary agent
all_tools = restaurant_tools