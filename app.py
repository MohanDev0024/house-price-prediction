import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Load Model & Features -------------------
model = joblib.load('house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# ------------------- Page Config -------------------
st.set_page_config(page_title="🏠 House Price Estimator", layout="wide")

# ------------------- Sidebar -------------------
st.sidebar.header("🏡 Enter House Features")

OverallQual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
GarageCars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
FullBath = st.sidebar.slider("Number of Full Bathrooms", 0, 4, 2)

# ------------------- Header -------------------
st.markdown("<h1 style='text-align: center;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict the selling price of a house based on key features</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- Predict Single Input -------------------
input_data = {
    'OverallQual': OverallQual,
    'GrLivArea': GrLivArea,
    'GarageCars': GarageCars,
    'TotalBsmtSF': TotalBsmtSF,
    'FullBath': FullBath
}

input_df = pd.DataFrame([input_data])
full_input = pd.DataFrame(columns=feature_names)
full_input = pd.concat([full_input, input_df], ignore_index=True)
full_input = full_input.fillna(0)

if st.button("🔍 Predict Price"):
    prediction = model.predict(full_input)[0]

    st.markdown(f"""
        <div style='text-align: center; padding: 20px; border-radius: 12px;
                    background-color: #f0f2f6; box-shadow: 2px 2px 8px #ccc;'>
            <h2>💰 Estimated Price</h2>
            <h1 style='color: green;'>${prediction:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)

# ------------------- Batch Prediction -------------------
st.markdown("---")
st.subheader("📁 Predict Prices from a CSV File")

uploaded_file = st.file_uploader("Upload a CSV file with the same feature columns:", type=['csv'])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    batch_input = pd.DataFrame(columns=feature_names)
    batch_input = pd.concat([batch_input, batch_df], ignore_index=True).fillna(0)

    preds = model.predict(batch_input)
    results = batch_df.copy()
    results['PredictedPrice'] = preds

    st.success("✅ Prediction complete. Preview below:")
    st.dataframe(results.head())

    # Download link
    csv = results.to_csv(index=False).encode()
    st.download_button("⬇️ Download Full Results", data=csv, file_name="predicted_prices.csv", mime='text/csv')

# ------------------- Feature Importance -------------------
def plot_feature_importance(model, features):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = model.coef_
        else:
            st.warning("Feature importances not available for this model.")
            return

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title('Top 5 Important Features')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to plot feature importance: {e}")

if st.checkbox("📊 Show Feature Importance"):
    plot_feature_importance(model, feature_names)

# ------------------- About -------------------
with st.expander("ℹ️ About this App"):
    st.markdown("""
    - This app predicts house prices using a trained Machine Learning model.
    - Built with: **Linear/Ridge/Lasso/Random Forest**, **Pandas**, **Scikit-learn**, **Streamlit**.
    - Supports both **single input prediction** and **bulk prediction via CSV upload**.
    - Ideal for learning, demo, or client pitch purposes.

    **Author**: Mohan Dev  
    💼 GitHub: [github.com/mohandev](https://github.com/mohandev)
    """)

