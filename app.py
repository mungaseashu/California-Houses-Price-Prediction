import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model and pipeline
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model.pkl")
        pipeline = joblib.load("pipeline.pkl")
        return model, pipeline
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'model.pkl' and 'pipeline.pkl' are in the same directory.")
        st.stop()

model, pipeline = load_models()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏡 California Housing Price Predictor</h1>
    <p>Predict median house values using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

# Info section
with st.container():
    st.markdown("""
    <div class="info-card">
        <h3>📊 How it works</h3>
        <p>This model uses location, housing characteristics, and demographic data to predict median house values in California. 
        Adjust the parameters below to see how different factors affect housing prices.</p>
    </div>
    """, unsafe_allow_html=True)

# Main layout with columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗺️ Location Parameters")
    
    # Location inputs with better formatting
    longitude = st.slider(
        "Longitude", 
        min_value=-125.0, 
        max_value=-114.0, 
        value=-118.0, 
        step=0.01,
        help="Geographic longitude coordinate"
    )
    
    latitude = st.slider(
        "Latitude", 
        min_value=32.0, 
        max_value=42.0, 
        value=34.0, 
        step=0.01,
        help="Geographic latitude coordinate"
    )
    
    ocean_proximity = st.selectbox(
        "Ocean Proximity", 
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        help="Proximity to ocean or bay"
    )
    
    st.subheader("🏠 Housing Characteristics")
    
    housing_median_age = st.slider(
        "Median Age of Houses", 
        min_value=1, 
        max_value=52, 
        value=20,
        help="Median age of houses in the block"
    )
    
    total_rooms = st.number_input(
        "Total Rooms", 
        min_value=1, 
        max_value=20000, 
        value=2000,
        step=100,
        help="Total number of rooms in the block"
    )

with col2:
    st.subheader("👥 Demographics")
    
    population = st.number_input(
        "Population", 
        min_value=1, 
        max_value=20000, 
        value=1000,
        step=50,
        help="Total population in the block"
    )
    
    households = st.number_input(
        "Households", 
        min_value=1, 
        max_value=5000, 
        value=400,
        step=25,
        help="Number of households in the block"
    )
    
    median_income = st.slider(
        "Median Income (in $10k)", 
        min_value=0.5, 
        max_value=15.0, 
        value=3.5, 
        step=0.1,
        help="Median household income (in tens of thousands of dollars)"
    )
    
    st.subheader("🛏️ Room Details")
    
    total_bedrooms = st.number_input(
        "Total Bedrooms", 
        min_value=1, 
        max_value=5000, 
        value=500,
        step=25,
        help="Total number of bedrooms in the block"
    )

# Prediction section
st.markdown("---")

# Center the predict button
col_center = st.columns([1, 2, 1])
with col_center[1]:
    predict_button = st.button("🔮 Predict House Value", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Analyzing housing data..."):
        # Create input dataframe
        input_df = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }])

        # Make prediction
        try:
            transformed_input = pipeline.transform(input_df)
            prediction = model.predict(transformed_input)
            predicted_value = prediction[0]
            
            # Display result with modern styling
            st.markdown(f"""
            <div class="prediction-result">
                🏠 Estimated Median House Value<br>
                <span style="font-size: 2.5rem;">${predicted_value:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Price per Room</h4>
                    <h3>${predicted_value/total_rooms:.0f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Price per Bedroom</h4>
                    <h3>${predicted_value/total_bedrooms:.0f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                income_ratio = predicted_value / (median_income * 10000)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Price-to-Income</h4>
                    <h3>{income_ratio:.1f}x</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                price_per_person = predicted_value / population
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Price per Person</h4>
                    <h3>${price_per_person:.0f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance visualization (if available)
            st.subheader("📈 Key Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.info(f"🌊 **Ocean Proximity**: {ocean_proximity}")
                st.info(f"💰 **Income Level**: ${median_income * 10:.0f}k (median)")
                
            with insights_col2:
                st.info(f"🏠 **Housing Age**: {housing_median_age} years (median)")
                st.info(f"👥 **Population Density**: {population/households:.1f} people/household")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>💡 <strong>Tip:</strong> Try adjusting the median income and ocean proximity to see how they impact housing prices!</p>
    <p><small>Model trained on California housing dataset • Predictions are estimates only</small></p>
</div>
""", unsafe_allow_html=True)