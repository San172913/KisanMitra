import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="KisanMitra", page_icon="🌱", layout="wide")
st.title("🌱 KisanMitra")
st.write("A simple crop recommendation system using soil and weather values.")

model_path = Path("models/best_model.joblib")
data_path = Path("data/Crop_recommendation.csv")
if not model_path.exists():
    st.warning("Model not found. First run setup_windows.bat to install packages and train the models.")
    st.stop()

model = joblib.load(model_path)
try:
    best_name = Path("models/best_model.txt").read_text()
except Exception:
    best_name = "Trained model"
st.caption("Selected model: " + best_name)

page = st.sidebar.radio("Choose page", ["Recommend Crop", "Model Analysis", "Dataset"])

if page == "Recommend Crop":
    st.subheader("Enter soil and weather details")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
        p = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
        k = st.number_input("Potassium (K)", 0.0, 250.0, 50.0)
    with c2:
        temperature = st.number_input("Temperature °C", -10.0, 60.0, 25.0)
        humidity = st.number_input("Humidity %", 0.0, 100.0, 70.0)
    with c3:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 100.0)
    if st.button("Recommend Crop", type="primary"):
        values = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        prediction = model.predict(values)[0]
        st.success("Recommended crop: " + str(prediction.title()))
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(values)[0]
            crops = model.classes_
            top = sorted(zip(crops, probabilities), key=lambda x: x[1], reverse=True)[:5]
            st.subheader("Top 5 model predictions")
            st.dataframe(pd.DataFrame(top, columns=["Crop", "Model score"]).assign(**{"Model score": lambda x: (x["Model score"] * 100).round(2).astype(str) + "%"}), use_container_width=True)
        st.caption("This is a machine-learning prediction, not a substitute for local agricultural advice.")

elif page == "Model Analysis":
    results_path = Path("outputs/model_results.csv")
    if results_path.exists():
        results = pd.read_csv(results_path)
        st.subheader("Four-model comparison")
        st.dataframe(results.round(3), use_container_width=True)
        if Path("outputs/model_comparison.png").exists():
            st.image("outputs/model_comparison.png", use_container_width=True)
    else:
        st.info("Run train_model.py first to create model results.")

else:
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.write("Dataset shape:", df.shape)
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Crop counts")
        st.bar_chart(df["label"].value_counts())
    else:
        st.info("Dataset CSV is missing. Add Crop_recommendation.csv inside the data folder.")

st.divider()
st.caption("KisanMitra student ML project | Uses N, P, K, temperature, humidity, pH and rainfall.")
