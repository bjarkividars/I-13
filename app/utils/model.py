
        
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_and_predict(df, input_features):
    """
    Trains a model and predicts the price for the given input features.
    """
    # One-hot encode zip_code if present
    feature_cols = list(input_features.keys())
    if 'zip_code' in feature_cols:
        df = pd.get_dummies(df, columns=['zip_code'], prefix='zip')
        feature_cols = [col for col in df.columns if col.startswith('zip_') or col in feature_cols]

        # Add one-hot encoding for the input features
        zip_code_col = f"zip_{input_features['zip_code']}"
        for col in feature_cols:
            if col.startswith("zip_") and col not in input_features:
                input_features[col] = 0
        input_features[zip_code_col] = 1

    # Drop rows with missing values in selected features
    clean_df = df.dropna(subset=feature_cols)

    # Split into features and target
    X = clean_df[feature_cols]
    y = clean_df['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test predictions
    predictions = model.predict(X_test)

    errors = np.abs(y_test - predictions)
    mean_absolute_percentage_error = np.mean(100 * (errors / y_test))

    # Predict for input features
    input_data = pd.DataFrame([input_features])
    input_prediction = model.predict(input_data[feature_cols])[0]

    return {
        'predicted_price': input_prediction,
        'confidence_mape': mean_absolute_percentage_error
    }
