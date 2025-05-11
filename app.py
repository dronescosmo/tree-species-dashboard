import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Tree Species Identifier", layout="wide")
st.title("🌳 Tree Species Identification Dashboard")

@st.cache_resource
def train_model(df):
    X = df.drop(columns=["label", "x", "y"])
    y = df["label"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    return model, le

# Sample training data (20 rows)
sample_df = pd.DataFrame({
    'z_mean': [14.5, 20.3, 12.9, 18.1, 22.3, 11.5, 15.0, 19.8, 21.1, 13.7]*2,
    'z_max': [23.4, 29.1, 20.8, 27.5, 31.2, 19.7, 24.5, 28.4, 30.0, 21.3]*2,
    'z_std': [2.1, 3.4, 1.8, 2.5, 3.0, 1.6, 2.2, 2.8, 3.1, 2.0]*2,
    'point_density': [0.6, 0.7, 0.5, 0.65, 0.75, 0.55, 0.62, 0.7, 0.77, 0.6]*2,
    'r_mean': [120, 135, 110, 130, 140, 105, 125, 138, 145, 115]*2,
    'g_mean': [115, 130, 100, 120, 135, 95, 118, 132, 138, 108]*2,
    'b_mean': [110, 125, 95, 115, 130, 90, 112, 128, 134, 105]*2,
    'x': [77.11 + i*0.001 for i in range(20)],
    'y': [28.55 + i*0.001 for i in range(20)],
    'label': ['Teak', 'Sal', 'Pine', 'Teak', 'Sal', 'Pine', 'Teak', 'Sal', 'Pine', 'Teak']*2
})

model, encoder = train_model(sample_df)

uploaded_file = st.file_uploader("📤 Upload tree_features.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'x' in df.columns and 'y' in df.columns:
        try:
            features = df.drop(columns=[col for col in ['label', 'x', 'y'] if col in df.columns])
            preds = encoder.inverse_transform(model.predict(features))
            df["predicted_species"] = preds

            st.success("✅ Prediction complete!")
            st.map(df.rename(columns={"y": "lat", "x": "lon"}))
            st.dataframe(df[["x", "y", "predicted_species"]].head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "predicted_species.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("CSV must contain 'x' and 'y' columns.")
