import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_artifacts():
    return joblib.load("house_model.pkl")


art = load_artifacts()
defaults = art["defaults"]

st.title("🏠 House Price Predictor")
st.write(f"Enter the house features to get an estimated price. Model: **{art['model_name']}**")

col1, col2 = st.columns(2)

with col1:
    neighborhood = st.selectbox("Neighborhood", art["neighborhoods"],
                                index=art["neighborhoods"].index("NAmes") if "NAmes" in art["neighborhoods"] else 0)
    overall_qual = st.slider("Overall quality (1-10)", 1, 10, 6)
    year_built   = st.number_input("Year built", 1870, 2010, 1975)
    year_remod   = st.number_input("Year remodeled", 1950, 2010, 1995)
    year_sold    = st.number_input("Year sold", 2006, 2010, 2008)
    lot_area     = st.number_input("Lot area (sq ft)", 1000, 100000, 9500, step=100)
    bedrooms     = st.number_input("Bedrooms above ground", 0, 10, 3)

with col2:
    gr_liv_area  = st.number_input("Above-ground living area (sq ft)", 300, 6000, 1500, step=10)
    bsmt_sf      = st.number_input("Basement area (sq ft)", 0, 5000, 1000, step=10)
    first_flr    = st.number_input("1st floor area (sq ft)", 300, 5000, 1100, step=10)
    second_flr   = st.number_input("2nd floor area (sq ft)", 0, 3000, 400, step=10)
    garage_cars  = st.slider("Garage capacity (cars)", 0, 4, 2)
    garage_area  = st.number_input("Garage area (sq ft)", 0, 1500, 480, step=10)
    full_bath    = st.slider("Full bathrooms", 0, 4, 2)
    half_bath    = st.slider("Half bathrooms", 0, 2, 1)

if st.button("Predict price"):
    row = pd.DataFrame([defaults])[art["columns"]]

    # Features saisies par l'utilisateur
    row.loc[0, "OverallQual"]  = overall_qual
    row.loc[0, "YearBuilt"]    = year_built
    row.loc[0, "YearRemodAdd"] = year_remod
    row.loc[0, "YrSold"]       = year_sold
    row.loc[0, "LotArea"]      = lot_area
    row.loc[0, "BedroomAbvGr"] = bedrooms
    row.loc[0, "GrLivArea"]    = gr_liv_area
    row.loc[0, "TotalBsmtSF"]  = bsmt_sf
    row.loc[0, "1stFlrSF"]     = first_flr
    row.loc[0, "2ndFlrSF"]     = second_flr
    row.loc[0, "GarageCars"]   = garage_cars
    row.loc[0, "GarageArea"]   = garage_area
    row.loc[0, "FullBath"]     = full_bath
    row.loc[0, "HalfBath"]     = half_bath

    # Quartier (one-hot) : on remet tout à 0 puis on active le bon
    for c in art["columns"]:
        if c.startswith("Neighborhood_"):
            row.loc[0, c] = 0
    if f"Neighborhood_{neighborhood}" in row.columns:
        row.loc[0, f"Neighborhood_{neighborhood}"] = 1

    # Features créées en Partie 2
    row.loc[0, "TotalSF"]   = bsmt_sf + first_flr + second_flr
    row.loc[0, "HouseAge"]  = year_sold - year_built
    row.loc[0, "RemodAge"]  = year_sold - year_remod
    row.loc[0, "TotalBath"] = (full_bath + 0.5 * half_bath
                               + row.loc[0, "BsmtFullBath"] + 0.5 * row.loc[0, "BsmtHalfBath"])
    row.loc[0, "HasGarage"] = int(garage_cars > 0 or garage_area > 0)

    pred_log = art["model"].predict(art["scaler"].transform(row))[0]
    price = float(np.expm1(pred_log))

    st.success(f"Estimated price: ${price:,.0f}")
