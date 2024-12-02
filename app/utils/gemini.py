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
            Focus on trying to find the following data in the requests: {columns_str} 
            and return only those found in a dict format exactly like this, no added quotation marks or anything:
            {{col1: value, col2: value, ... ,}}.
            If none of the relevant columns are found in the requests found, return only "N/A", that is only if none are found, make sure you are 100% certain that none are present before returning "N/A".
            """
        )

    def process_request(self, text: str):
        if not self.model:
            raise Exception("Model not initialized. Please call setup_model first.")
        
        response = self.model.generate_content(f"Extract data for this request: {text}")
        print(f'RESPONSE {response.text}')
        return response.text