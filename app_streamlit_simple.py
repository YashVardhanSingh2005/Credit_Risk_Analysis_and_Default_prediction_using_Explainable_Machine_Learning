# app_streamlit_simple.py
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt

# ------------------ CONFIG: change these paths ------------------
#PREPROCESSED_CSV_PATH = "/content/drive/MyDrive/lending_club_preprocessed.csv"
#MODEL_PATH = "/content/drive/MyDrive/lgbm_tuned_final.pkl"

# UPDATED FOR DEPLOYMENT
PREPROCESSED_CSV_PATH = "app_sample_data.csv"
MODEL_PATH = "lgbm_tuned_final.pkl"
# ---------------------------------------------------------------

st.set_page_config(page_title="Loan PD Demo (LightGBM + SHAP + Threshold)", layout="wide")

@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError("preprocessed CSV must include a 'target' column")
    features = df.drop(columns=["target"]).columns.tolist()
    medians = df[features].median(numeric_only=True)
    bool_cols = df[features].select_dtypes(include=["bool","uint8","int8"]).columns.tolist()
    return df, features, medians.to_dict(), bool_cols

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Load resources once
with st.spinner("Loading model and dataset..."):
    df_all, FEATURE_COLS, medians_dict, BOOL_COLS = load_data(PREPROCESSED_CSV_PATH)
    MODEL = load_model(MODEL_PATH)

# ---------------- SIDEBAR: threshold presets & input UI ----------------
st.sidebar.header("Controls")

# Threshold presets & slider
preset = st.sidebar.selectbox("Threshold preset", [
    "Best-F1 (recommended)", "Recall ≈ 0.60", "Default 0.50", "Custom"
])
PRESET_MAP = {
    "Best-F1 (recommended)": 0.285,
    "Recall ≈ 0.60": 0.29,
    "Default 0.50": 0.50
}
if preset != "Custom":
    threshold = float(PRESET_MAP[preset])
    st.sidebar.write(f"Using threshold = {threshold:.3f}")
else:
    threshold = float(st.sidebar.slider("Custom threshold", 0.0, 1.0, 0.29, 0.005))

st.sidebar.markdown("**Note:** For imbalanced credit data 0.5 is often too high — use lower threshold like Best-F1 to balance precision & recall.")

st.sidebar.markdown("---")
st.sidebar.header("Input features (defaults = medians)")

# Build form
form = st.sidebar.form("input_form")
input_values = {}
num_cols = []
bool_cols = []

for c in FEATURE_COLS:
    if c in BOOL_COLS:
        bool_cols.append(c)
    else:
        num_cols.append(c)

# numeric inputs
for c in num_cols:
    default = float(medians_dict.get(c, 0.0))
    input_values[c] = form.number_input(label=c, value=default, format="%.6f")

# boolean inputs
if bool_cols:
    for c in bool_cols:
        default_bool = bool(medians_dict.get(c, 0))
        input_values[c] = int(form.checkbox(label=c, value=default_bool))

submit = form.form_submit_button("Predict")

# ---------------- Main: prediction + SHAP ----------------
if submit:
    try:
        # Build input row (ordered exactly as model expects)
        X_row = pd.DataFrame([input_values])[FEATURE_COLS].astype(float)

        # Predict probability (handles sklearn wrapper or raw booster)
        if hasattr(MODEL, "predict_proba"):
            proba = float(MODEL.predict_proba(X_row)[:, 1][0])
        else:
            proba = float(MODEL.predict(X_row)[0])

        # Decision with chosen threshold
        decision = "REJECT (High risk)" if proba >= threshold else "APPROVE (Low risk)"
        st.metric("Predicted Probability of Default (PD)", f"{proba:.4f}")
        if "REJECT" in decision:
            st.error(f"Decision (thr={threshold:.3f}): {decision}")
        else:
            st.success(f"Decision (thr={threshold:.3f}): {decision}")

        # Show input snapshot
        with st.expander("Show input values"):
            st.dataframe(X_row.T)

        # SHAP explanation (robust to multiple shap versions)
        st.subheader("SHAP explanation — top contributors")
        try:
            if hasattr(MODEL, "booster_"):
                explainer = shap.TreeExplainer(MODEL.booster_)
            else:
                explainer = shap.TreeExplainer(MODEL)
            shap_values = explainer.shap_values(X_row)

            # handle list-of-classes vs single-array outputs
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = shap_values[1][0]    # class-1 shap values
                base_value = float(explainer.expected_value[1])
            else:
                if hasattr(shap_values, "ndim") and shap_values.ndim == 2:
                    sv = shap_values[0]
                else:
                    sv = shap_values
                ev = explainer.expected_value
                base_value = float(ev[1]) if isinstance(ev, (list, tuple, np.ndarray)) and len(ev) > 1 else float(ev)

            sv = np.array(sv).reshape(-1)
            feat_names = FEATURE_COLS

            df_shap = pd.DataFrame({
                "feature": feat_names,
                "feature_value": [float(X_row.iloc[0][f]) for f in feat_names],
                "shap_value": sv
            })

            top_pos = df_shap[df_shap["shap_value"] > 0].sort_values("shap_value", ascending=False).head(7)
            top_neg = df_shap[df_shap["shap_value"] < 0].sort_values("shap_value").head(7)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top features increasing PD**")
                st.dataframe(top_pos.reset_index(drop=True))
            with col2:
                st.markdown("**Top features reducing PD**")
                st.dataframe(top_neg.reset_index(drop=True))

            # Plot top absolute contributors
            df_shap["abs_shap"] = df_shap["shap_value"].abs()
            top_abs = df_shap.sort_values("abs_shap", ascending=False).head(12).set_index("feature")
            colors = ["red" if v > 0 else "green" for v in top_abs["shap_value"]]
            fig, ax = plt.subplots(figsize=(8, 5))
            top_abs["shap_value"].plot(kind="barh", ax=ax, color=colors)
            ax.set_xlabel("SHAP value (impact on PD)")
            ax.invert_yaxis()
            st.pyplot(fig, use_container_width=True)

            st.write(f"Model base value (expected): {base_value:.4f}")
        except Exception as e:
            st.error("SHAP explanation failed: " + str(e))

    except Exception as e:
        st.error("Prediction failed: " + str(e))
