import streamlit as st
import pandas as pd
import joblib
import numpy as np

from src.config import MODELS_DIR, TIER_LABELS, TRAIN_CSV, USE_LOG_TARGET

# Load Models and Base Data
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load(MODELS_DIR / "random_forest_reg.joblib")
        clf_model = joblib.load(MODELS_DIR / "random_forest_clf.joblib")
        return reg_model, clf_model
    except FileNotFoundError:
        st.error("Model files not found. Please run `python run_pipeline.py` first.")
        st.stop()

@st.cache_data
def load_template_house():
    try:
        df = pd.read_csv(TRAIN_CSV)
        df = df.drop(columns=["Id", "SalePrice"], errors="ignore")
        
        template = pd.DataFrame([df.median(numeric_only=True)])
        for col in df.select_dtypes(include=['object']).columns:
            template[col] = df[col].mode()[0]
            
        return template
    except FileNotFoundError:
        st.error("train.csv not found in data/kaggle_data/. Please download it.")
        st.stop()

reg_model, clf_model = load_models()
template_house = load_template_house()

# Streamlit UI
st.title("Real Estate Price Predictor")
st.write("Enter the details of a home in Ames, Iowa to predict its sale price and market tier.")

st.markdown("### House Features")
col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
    year_built = st.number_input("Year Built", min_value=1872, max_value=2010, value=2000)
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=6)

with col2:
    neighborhood = st.selectbox("Neighborhood", options=[
        'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
        'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
        'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
        'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
    ])
    bldg_type = st.selectbox("Building Type", options=['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'])

# Make Predictions
if st.button("Predict Price", type="primary"):
    input_data = template_house.copy()
    
    input_data['GrLivArea'] = gr_liv_area
    input_data['YearBuilt'] = year_built
    input_data['OverallQual'] = overall_qual
    input_data['Neighborhood'] = neighborhood
    input_data['BldgType'] = bldg_type
    
    raw_pred = reg_model.predict(input_data)[0]
    final_price = np.expm1(raw_pred) if USE_LOG_TARGET else raw_pred
    
    tier_idx = clf_model.predict(input_data)[0]
    tier_label = TIER_LABELS[tier_idx]
    
    st.divider()
    st.subheader("Results")
    st.metric(label="Predicted Sale Price", value=f"${final_price:,.2f}")
    st.metric(label="Market Tier", value=tier_label)