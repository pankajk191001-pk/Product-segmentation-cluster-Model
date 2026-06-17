import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Product Segmentation", layout="centered")

# -----------------------------
# Load and train model
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("realistic_e_commerce_sales_data.csv")

    numeric_features = [
        "Unit Price",
        "Quantity",
        "Total Price",
        "Shipping Fee",
        "Age"
    ]

    categorical_features = [
        "Category",
        "Gender",
        "Product Name",
        "Region"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(n_clusters=4, random_state=42))
    ])

    X_train = df[numeric_features + categorical_features]
    pipeline.fit(X_train)

    return pipeline, df


pipeline, df = train_model()

cluster_mapping = {
    0: "Luxury & High-End Products",
    1: "Low-Cost Daily Essentials",
    2: "Popular Mid-Range Products",
    3: "Niche or Impulse Products"
}

# -----------------------------
# UI
# -----------------------------
st.title("Product Data Segmentation with KMeans")

unit_price = st.number_input("Unit Price", min_value=0.0, value=10.0)
quantity = st.number_input("Quantity", min_value=1, value=1)
total_price = st.number_input("Total Price", min_value=0.0, value=10.0)
shipping_fee = st.number_input("Shipping Fee", min_value=0.0, value=2.0)
age = st.number_input("Age", min_value=0, value=30)

category = st.selectbox(
    "Category",
    sorted(df["Category"].dropna().unique())
)

gender = st.selectbox(
    "Gender",
    sorted(df["Gender"].dropna().unique())
)

product_name = st.selectbox(
    "Product Name",
    sorted(df["Product Name"].dropna().unique())
)

region = st.selectbox(
    "Region",
    sorted(df["Region"].dropna().unique())
)

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

    st.success(
        f"Predicted Cluster: {cluster} - {cluster_mapping.get(cluster, 'Unknown')}"
    )
