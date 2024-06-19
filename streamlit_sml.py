import streamlit as st
import numpy as np
import pickle
import os

model_paths = {
    'SVM_water_area': 'SVM_water_area.pkl',
    'SVM_vegetation_area': 'SVM_vegetation_area.pkl',
    'SVM_urban_area': 'SVM_urban_area.pkl',
    'Gradient_Boosting_water_area': 'Gradient_Boosting_water_area.pkl',
    'Gradient_Boosting_vegetation_area': 'Gradient_Boosting_vegetation_area.pkl',
    'Gradient_Boosting_urban_area': 'Gradient_Boosting_urban_area.pkl',
    'Random_Forest_water_area': 'Random_Forest_water_area.pkl',
    'Random_Forest_vegetation_area': 'Random_Forest_vegetation_area.pkl',
    'Random_Forest_urban_area': 'Random_Forest_urban_area.pkl',
    'Linear_Regression_water_area': 'Linear_Regression_water_area.pkl',
    'Linear_Regression_vegetation_area': 'Linear_Regression_vegetation_area.pkl',
    'Linear_Regression_urban_area': 'Linear_Regression_urban_area.pkl'
}

# Load models directly
models = {key: pickle.load(open(path, 'rb')) for key, path in model_paths.items()}

# Hardcoded average densities for each month
monthly_avg_densities = {
    1: {'vegetation_density': 0.2756, 'water_density': 0.01454, 'urban_density': 0.57678},
    2: {'vegetation_density': 0.26108, 'water_density': 0.05166, 'urban_density': 0.56644},
    3: {'vegetation_density': 0.1622, 'water_density': 0.00606, 'urban_density': 0.67652},
    4: {'vegetation_density': 0.2195, 'water_density': 0.00538, 'urban_density': 0.65786},
    5: {'vegetation_density': 0.20626, 'water_density': 0.0094, 'urban_density': 0.67592},
    6: {'vegetation_density': 0.743525, 'water_density': 0.014846, 'urban_density': 0.239436},
    7: {'vegetation_density': 0.719823, 'water_density': 0.014847, 'urban_density': 0.198218},
    8: {'vegetation_density': 0.654542, 'water_density': 0.014441, 'urban_density': 0.201167},
    9: {'vegetation_density': 0.760818, 'water_density': 0.0147660, 'urban_density': 0.227956},
    10: {'vegetation_density': 0.732601, 'water_density': 0.0148583, 'urban_density': 0.21654},
    11: {'vegetation_density': 0.49782, 'water_density': 0.01556, 'urban_density': 0.40468},
    12: {'vegetation_density': 0.35042, 'water_density': 0.01076, 'urban_density': 0.51628}
}

def get_model_and_densities(month, model_type, area_type):
    # Notice the removal of '_area.pkl' to match the actual keys in the models dictionary
    model_filename = f"{model_type}_{area_type}_area"
    model = models.get(model_filename)
    avg_densities = monthly_avg_densities.get(month, {'vegetation_density': 0.5, 'water_density': 0.2, 'urban_density': 0.3})
    return model, avg_densities


# Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ('Home', 'Trend', 'Prediction'))

if section == 'Home':
    st.header("Change Analysis in Pune")
    st.write("""
        This project aims to provide insights into the environmental and urban changes in Pune over recent years. Through detailed analysis and predictive modeling, we explore trends, predict future changes, and understand the impact of human activities on urban areas.
        
        Understanding the dynamics of urban expansion and environmental impact requires robust data analysis and predictive tools to aid city planners and policy makers in making informed decisions.
    """)
    st.image("Screenshot 2024-04-15 041025.png", caption="Data Labelling Illustration")

elif section == 'Trend':
    st.header("Trend Analysis Dashboard")
    trend_option = st.selectbox("Choose a trend to display", ('Vegetation', 'Water', 'Urban', 'Overall'))
    
    if trend_option == 'Vegetation':
        st.image("Vegetation_Area_Trends.png", caption="Vegetation Trends")
        st.write("""
           Original: High variability, with the largest values among the three categories.\n
           Trend: Starts relatively stable, then declines from 2021 to 2023, potentially indicating a reduction in vegetation areas.\n
            Seasonality: Very pronounced, indicating strong seasonal effects on vegetation, likely due to natural growth cycles.\n
            Residuals: Notable spikes, especially around 2021 and 2023, suggesting events or changes not captured by the model.
        """)
    elif trend_option == 'Water':
        st.image("Water_Area_Trends.png", caption="Water Trends")
        st.write("""
             Original: Overall lower values than urban areas with similar variability.\n
             Trend: Shows an increasing trend until mid-2021, followed by a sharp decline, leveling off in 2023.\n
             Seasonality: Noticeable seasonality, but less pronounced than in urban areas.\n
             Residuals: Larger spikes, particularly in 2021, indicate some significant anomalies not explained by trend or seasonality.
        """)
    elif trend_option == 'Urban':
        st.image("Urban_Area_Trends.png", caption="Urban Trends")
        st.write("""
            Original: Fluctuations with several peaks and troughs suggest variability in urban area data points.
            Trend: A general upward trend, implying an increase in urban areas over time.
            Seasonality: Strong, consistent seasonal patterns, possibly indicating seasonal changes in urban activities or reporting.
            Residuals: Residuals are relatively small, which suggests that the trend and seasonality components capture most of the variation in the data.
        """)
    elif trend_option == 'Overall':
        
        st.write("""
            The data indicates expansion in urban areas, with a strong seasonal component, suggesting that urban development may be cyclical or impacted by seasonal factors. Water areas fluctuated with a significant shift in mid-2021, which could be due to environmental events or changes in measurement. Vegetation shows the most pronounced seasonality, consistent with natural growth and dormancy cycles, but the overall decline in trend raises concerns about possible deforestation or land conversion. The residuals in all three graphs highlight the presence of anomalies or irregularities that the trend and seasonal adjustments do not fully account for, indicating complex dynamics at play. The year 2021 appears to be particularly anomalous across all three categories, suggesting an event or series of events that significantly impacted these areas.
        """)

elif section == 'Prediction':
    st.header("Prediction Interface")
    month = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: f"Month {x}")
    model_type = st.selectbox("Select Model Type", ['Linear_Regression', 'Random_Forest', 'Gradient_Boosting', 'SVM'])
    area_type = st.selectbox("Select Area Type", ['vegetation', 'water', 'urban'])
    
    predict_button = st.button("Predict")
    if predict_button:
        model, avg_densities = get_model_and_densities(month, model_type, area_type)
        if not model:
            st.error("Model not found.")
        else:
            year = 2024  # As per your fixed year setup
            inputs = np.array([[year, month, avg_densities['vegetation_density'], avg_densities['water_density'], avg_densities['urban_density']]])
            prediction = model.predict(inputs)[0]
            st.success(f"Prediction: {prediction}")
            st.image("Screenshot 2024-04-15 0347511.png", caption="Accuracy detail")


