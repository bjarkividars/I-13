import locale
import os
import gdown
import streamlit as st
import pandas as pd
from utils.gemini import ModelHandler
from utils.data_preprocessing import clean_data, preprocess_data
from utils.model import train_and_predict
from utils.text_utils import extract_features_from_text
from dotenv import load_dotenv
import plotly.express as px

csv_url = 'https://drive.google.com/uc?id=1JiJctVOQi_feJIeHjONy7F_mh2wNv0u7&export=download'
output = 'realtor-data.csv'
if not os.path.exists(output):
    print(f"File not found. Downloading {output}...")
    gdown.download(csv_url, output, quiet=False)

primaryColor = "#008ECC"
secondaryColor = "#F0F8FF"


def on_city_change():
    """Update filtered data when city selection changes."""
    if 'cleaned_data' in st.session_state and 'valid_cities' in st.session_state:
        selected_city_state = st.session_state.selected_city
        if selected_city_state is not None:
            # Update filtered data
            st.session_state.filtered_data = preprocess_data(
                st.session_state.cleaned_data, selected_city_state
            )
        else:
            # Remove only the 'filtered_data' key from session state
            if 'filtered_data' in st.session_state:
                del st.session_state.filtered_data


st.set_page_config(page_title="Prophecy", layout="centered")

st.title("Real Estate Price Prediction")
st.write(
    "Use this app to predict real estate prices based on a city and a description of the property."
)

if 'cleaned_data' not in st.session_state or 'valid_cities' not in st.session_state:
    try:
        load_dotenv()
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        raw_df = pd.read_csv('realtor-data.csv')
        cleaned_data, valid_cities = clean_data(raw_df)
        x_columns = list(cleaned_data.columns)
        x_columns.remove('city')
        st.session_state.model_handler = ModelHandler(
            os.environ['GEMINI_API_KEY'], x_columns)
        st.session_state.cleaned_data = cleaned_data
        st.session_state.valid_cities = valid_cities
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

if 'cleaned_data' not in st.session_state or 'valid_cities' not in st.session_state:
    st.error("Failed to load and clean data.")
    st.stop()

# Use CityState objects in the selectbox
selected_city = st.selectbox(
    "Select a City",
    sorted(st.session_state.valid_cities,
           key=lambda x: x.city),  # Sort by city name
    format_func=str,  # Use the custom __str__ method for display
    key="selected_city",
    on_change=on_city_change,
    index=None
)

if 'filtered_data' in st.session_state and not st.session_state.filtered_data.empty:
    st.subheader("Price Distribution in Selected City")

    # Display a multiselect for zip code filtering
    zip_code_filter = st.multiselect(
        "Filter by Zip Code (Optional)",
        options=[int(option)
                 for option in st.session_state.filtered_data['zip_code'].unique()],
        default=None,
        help="Select one or more zip codes to filter the price distribution chart."
    )

    # Filter data based on selected zip codes
    filtered_data_by_zip = st.session_state.filtered_data
    if zip_code_filter:
        filtered_data_by_zip = filtered_data_by_zip[filtered_data_by_zip['zip_code'].isin(
            zip_code_filter)]

    # Create a distribution plot for house prices
    fig = px.histogram(
        filtered_data_by_zip,
        x="price",
        nbins=30,
        title="Distribution of House Prices",
        labels={"price": "Price (USD)"},
        template="plotly_white",
        color_discrete_sequence=[primaryColor],
    )

    # Update layout for colors
    fig.update_layout(
        title_font=dict(size=20, color=primaryColor),
        plot_bgcolor=secondaryColor,
        paper_bgcolor=secondaryColor,
        font=dict(color=primaryColor),
        xaxis_title="House Prices",
        yaxis_title="Count",
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Property Description")
property_description = st.text_area(
    "Enter a description of the property (size, bedrooms, bathrooms, zip code, etc.), or street and zip code:",
    placeholder="Example: A 2500 sq ft house with 4 bedrooms, 3 bathrooms, on a 0.5-acre lot in Zip Code 23220. Or 1600 Pennsylvania Ave 20500."
)

if st.button("Predict Price", type="primary"):
    if not property_description:
        st.error("Please enter a property description.")
    elif 'filtered_data' not in st.session_state or st.session_state.filtered_data.empty:
        st.error("Please select a city first.")
    else:
        try:
            extracted_features = extract_features_from_text(
                property_description, st.session_state.model_handler)

            if not extracted_features:
                st.subheader("Not Enough Data")
                st.write(
                    "Unable to make a prediction based on the input. Try adding any of the following:"
                )

                bullet_points = "\n".join(
                    [f"- {c.replace('_', ' ').capitalize()}" for c in st.session_state.cleaned_data.columns])
                st.markdown(bullet_points)

            else:
                if 'zestimate' in extracted_features.keys() and extracted_features['zestimate'] is not None:
                    zestimate = extracted_features['zestimate']
                    del extracted_features['zestimate']
                else:
                    zestimate = None

                if 'zip_code' in extracted_features.keys():
                    # Convert to float
                    zip_code = float(extracted_features['zip_code'])
                    # Ensure all are floats
                    u_zip_code = [
                        float(u) for u in st.session_state.filtered_data['zip_code'].unique()]

                    if zip_code not in u_zip_code:
                        del extracted_features['zip_code']
                        st.write(f"""**Zip Code Not Used In Model:** The zip code {
                                 zip_code} is not in the training data for this city, so it has been discarded.""")
                        st.write("**Available Zip Codes:**")
                        formatted_zip_codes = ", ".join(
                            [f"{int(zip_code)}" for zip_code in u_zip_code])
                        st.markdown(
                            f"""
                            <div style="
                                max-height: 150px;
                                overflow-y: scroll;
                                border: 2px solid #F0F8FF;
                                border-radius: 8px;
                                padding: 10px;">
                                {formatted_zip_codes}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                result = train_and_predict(
                    st.session_state.filtered_data, extracted_features
                )

                predicted_price = result['predicted_price']
                confidence_mape = result['confidence_mape']

                lower_bound = predicted_price * (1 - confidence_mape / 100)
                upper_bound = predicted_price * (1 + confidence_mape / 100)

                formatted_predicted_price = locale.format_string(
                    "%d", predicted_price, grouping=True)
                formatted_lower_bound = locale.format_string(
                    "%d", lower_bound, grouping=True)
                formatted_upper_bound = locale.format_string(
                    "%d", upper_bound, grouping=True)

                st.subheader("Prediction Results")
                st.write(f"**Predicted Price:** ${formatted_predicted_price}")
                st.write(f"""**Price Range (based on {int(confidence_mape)}% MAPE):** \\${
                         formatted_lower_bound} - \\${formatted_upper_bound}""")
                if (zestimate):
                    formatted_zestimate = locale.format_string(
                        "%d", float(zestimate), grouping=True)
                    st.write(f"**Zestimate:** ${formatted_zestimate}")
                else:
                    st.write(
                        'Try putting in your address to see a comparison with the Zestimate')

        except Exception as e:
            st.error(f"Error in prediction. Make sure all values are correct")
