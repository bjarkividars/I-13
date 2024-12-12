from typing import List, Tuple
import pandas as pd

class CityState:
    """Class to represent a city and state combination."""
    def __init__(self, city: str, state: str):
        self.city = city
        self.state = state

    def __str__(self):
        """Custom string representation for dropdown display."""
        return f"{self.city}, {self.state}"


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[CityState]]:
    """Clean data and return filtered DataFrame along with CityState objects."""
    df = df[df['status'] == 'sold']
    df = df.drop(columns=['brokered_by', 'status', 'street', 'prev_sold_date'])
    
    # Group by city and state and count occurrences
    df['city_state'] = df['city'] + ", " + df['state']  # Create a combined city-state column
    city_state_counts = df['city_state'].value_counts()

    # Filter valid city-state combinations with more than 1000 records
    valid_city_states = city_state_counts[city_state_counts > 1000].index
    fdf = df[df['city_state'].isin(valid_city_states)]
    fdf = fdf.reset_index(drop=True)

    # Create a list of CityState objects
    unique_city_states = fdf[['city', 'state']].drop_duplicates()
    city_state_objects = [
        CityState(row['city'], row['state']) for _, row in unique_city_states.iterrows()
    ]
    
    return fdf, city_state_objects

def preprocess_data(df, city_state):
    """
    Filters the data for the specified city and state and removes price outliers.
    """
    # Filter data by city and state
    df = df[(df['city'] == city_state.city) & (df['state'] == city_state.state)]

    # Remove outliers based on price percentiles
    lower_bound = df['price'].quantile(0.025)
    upper_bound = df['price'].quantile(0.975)
    df = df[(df['price'] > lower_bound) & (df['price'] < upper_bound)]

    return df

