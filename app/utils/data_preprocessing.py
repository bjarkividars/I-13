from typing import List, Tuple
import pandas as pd

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df[df['status'] == 'sold']
    df = df.drop(columns=['brokered_by', 'status', 'street', 'prev_sold_date', 'state'])
    city_counts = df['city'].value_counts()

    valid_cities = city_counts[city_counts > 5000].index
    fdf = df[df['city'].isin(valid_cities)]
    fdf = fdf.reset_index(drop=True)
    return fdf, valid_cities

def preprocess_data(df, city):
    """
    Filters the data for the specified city and removes price outliers.
    """
    # Filter data by city
    df = df[df['city'] == city]

    # Remove outliers based on price percentiles
    lower_bound = df['price'].quantile(0.025)
    upper_bound = df['price'].quantile(0.975)
    df = df[(df['price'] > lower_bound) & (df['price'] < upper_bound)]

    return df

