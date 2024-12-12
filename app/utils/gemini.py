from typing import List
import google.generativeai as genai
from typing import List
import os


class ModelHandler:
    def __init__(self, api_key: str, columns: List[str]):
        genai.configure(api_key=api_key)
        columns_str = ", ".join(columns)
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=f"""You are an extracting machine.
    Focus on identifying structured data in the requests.

    If an address is present, prioritize extracting its components (street, city, state, and zip) and return them in valid Python dictionary format. Include only the components that are explicitly present in the request; do not substitute or infer missing components. For example, if the city is not mentioned in the request, do not include anything about the city in the response. Return the dictionary exactly like this:
    {{'street': 'value', 'city': 'value', 'state': 'value', 'zip': 'value'}}. Use two-letter abbreviations for the state.

    If no address is found, attempt to find the following data in the requests: {columns_str}. Return the extracted values in valid Python dictionary format exactly like this:
    {{'col1': 'value', 'col2': 'value', ...}}.

    Return only the dictionary directly, without adding any prefixes like ```json or extra text. Ensure the output is valid Python dictionary syntax so it can be directly parsed in Python.

    If none of the relevant columns or address components are found, return only "N/A". Ensure you are 100% certain that none are present before returning "N/A".
    """
        )

    def process_request(self, text: str):
        if not self.model:
            raise Exception(
                "Model not initialized. Please call setup_model first.")

        response = self.model.generate_content(
            f"Extract data for this request: {text}")
        return response.text
