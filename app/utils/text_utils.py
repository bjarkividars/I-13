import json
import re
from typing import Dict
from utils.zillow import get_property_info
from utils.gemini import ModelHandler
import streamlit as st


def extract_json_content(text: str) -> str:
    """
    Extracts the content between the first '{' and the last '}' in the provided text.

    Parameters:
        text (str): The input text containing JSON content.

    Returns:
        str: The JSON string extracted from the input text.

    Raises:
        ValueError: If no JSON content is found in the text.
    """
    try:
        # Check if the input is already valid JSON
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # Use regex to extract JSON if the input is not directly valid JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_content = match.group(0)
            # Convert single quotes to double quotes to make it valid JSON
            json_content = json_content.replace("'", '"')
            return json_content
        else:
            raise ValueError("No JSON content found in the response text.")


def extract_features_from_text(description: str, model_handler: ModelHandler) -> Dict[str, float]:
    """
    Extract features from property description using the Gemini model and clean the response.

    Parameters:
        description (str): The property description provided by the user.
        model_handler (ModelHandler): The initialized Gemini model handler for extracting features.

    Returns:
        Dict[str, float]: A dictionary of cleaned features with numerical values.
    """
    # Use the Gemini model to process the text
    response = model_handler.process_request(description)

    if 'N/A' in response:
        return {}

    if 'street' in response:
        address_data = eval(response)
        address = address_data['street']
        city = st.session_state.selected_city.city
        state = st.session_state.selected_city.state
        zip_code = ''
        if 'zip' in address_data.keys():
            zip_code = address_data['zip']

        # Call the Zillow function with the extracted address
        zillow_response = get_property_info(address, city, state, zip_code)
        return zillow_response

    try:
        # Extract the JSON content from the response
        json_content = extract_json_content(response)

        # Parse the extracted JSON into a dictionary
        extracted_features = json.loads(json_content)

        # Clean and process the extracted features
        cleaned_features = {}
        for key, value in extracted_features.items():
            # Extract numerical part if value is a string
            if isinstance(value, str):
                # Find numbers in the string
                match = re.search(r'\d+(\.\d+)?', value)
                if match:
                    cleaned_features[key] = float(
                        match.group())  # Convert to float
                else:
                    # Assign None if no number found
                    cleaned_features[key] = None
            else:
                cleaned_features[key] = value  # Keep numeric values as is
        return cleaned_features

    except json.JSONDecodeError:
        raise ValueError(
            f"Failed to parse the extracted JSON content: {json_content}")
    except Exception as e:
        raise ValueError(f"Error processing the model's response: {e}")
