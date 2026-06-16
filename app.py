
import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("kmeans_pipeline.pkl")

# Cluster label mapping
cluster_mapping = {
    0: "Luxury & High-End Products",
    1: "Low-Cost Daily Essentials",
    2: "Popular Mid-Range Products",
    3: "Niche or Impulse Products"
}

st.title("product Data  Segmentation with KMeans")

# Input fields for all features used in training
unit_price = st.number_input("Unit Price", min_value=0.0, value=10.0)
quantity = st.number_input("Quantity", min_value=1, value=1)
total_price = st.number_input("Total Price", min_value=0.0, value=10.0)
shipping_fee = st.number_input("Shipping Fee", min_value=0.0, value=2.0)
age = st.number_input("Age", min_value=0, value=30)

category = st.text_input("Category", "Electronics")
gender = st.text_input("Gender", "Male")
product_name = st.text_input("Product Name", "Smartphone")
region = st.text_input("Region", "North")

# Predict button
if st.button("Predict Cluster"):
    input_df = pd.DataFrame([{
        "Unit Price": unit_price,
        "Quantity": quantity,
        "Total Price": total_price,
        "Shipping Fee": shipping_fee,
        "Age": age,
        "Category": category,
        "Gender": gender,
        "Product Name": product_name,
        "Region": region
    }])
    
    cluster = pipeline.predict(input_df)[0]
    
    st.success(f"Predicted Cluster: {cluster} - {cluster_mapping.get(cluster, 'Unknown')}")
