import http.client
import os
import urllib.parse
import json

import http.client
import urllib.parse
import json


def fetch_property_data(address, city, state, zip_code):
    """Base function to fetch property data from the API."""
    conn = http.client.HTTPSConnection("zillow-working-api.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': os.environ['GEMINI_API_KEY'],
        'x-rapidapi-host': "zillow-working-api.p.rapidapi.com"
    }

    # Construct the full address
    full_address = f"{address}, {city}, {state} {zip_code}"
    encoded_address = urllib.parse.quote(full_address)

    # Construct the API endpoint with the encoded address
    endpoint = f"/pro/byaddress?propertyaddress={encoded_address}"

    # Make the API request
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()

    # Decode the response
    try:
        response_json = json.loads(data.decode("utf-8"))
        return response_json
    except json.JSONDecodeError:
        return None


def get_property_zestimate(address, city, state, zip_code):
    """Fetch the Zestimate of the property."""
    property_data = fetch_property_data(address, city, state, zip_code)
    if not property_data:
        return "Failed to fetch property data"

    zestimate = property_data.get("propertyDetails", {}).get("zestimate")
    return zestimate if zestimate is not None else "Zestimate not available"


def get_property_info(address, city, state, zip_code):
    """Fetch general property information such as bedrooms, bathrooms, house size, and lot size."""
    property_data = fetch_property_data(address, city, state, zip_code)
    if not property_data:
        return "Failed to fetch property data"

    property_details = property_data.get("propertyDetails", {})
    result = {}

    bedrooms = property_details.get("bedrooms", "N/A")
    if bedrooms != "N/A":
        result["bed"] = bedrooms

    bathrooms = property_details.get("bathrooms", "N/A")
    if bathrooms != "N/A":
        result["bath"] = bathrooms

    house_size = property_details.get("livingArea", "N/A")
    if house_size != "N/A":
        result["house_size"] = house_size

    lot_size = property_details.get("lotAreaValue", "N/A")
    lot_units = property_details.get("lotAreaUnits", "N/A")
    if lot_units == "Square Feet" and lot_size != "N/A":
        # Convert lot size to acres (1 acre = 43,560 square feet)
        lot_size_acres = round(float(lot_size) / 43560, 2)
        result["acre_lot"] = lot_size_acres

    # Always include zip_code since it’s a required input
    result["zip_code"] = zip_code

    result['zestimate'] = get_property_zestimate(
        address, city, state, zip_code)
    return result
