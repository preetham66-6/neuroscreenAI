# =============================================================================
# FILE: app.py
# PURPOSE: Streamlit web app for Parkinson's detection using 16 voice features.
#          Three input methods:
#            1. Manual entry of 16 voice features
#            2. CSV file upload (16 columns)
#            3. WAV audio upload (features extracted automatically via Parselmouth)
# HOW TO RUN: python -m streamlit run app.py
# =============================================================================

import os
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

matplotlib.use("Agg")

import streamlit as st
from extract_features import extract_features


# =============================================================================
# SECTION 1: Page Configuration
# =============================================================================

st.set_page_config(
    page_title="NeuroScreen AI",
    page_icon="🧠",
    layout="wide"
)


# =============================================================================
# SECTION 2: Session State Initialisation
# Streamlit re-runs the whole script on every interaction.
# session_state lets us keep values (like extracted features) across re-runs.
# =============================================================================

if "wav_features" not in st.session_state:
    st.session_state.wav_features = None   # stores extracted feature dict

if "wav_error" not in st.session_state:
    st.session_state.wav_error = None      # stores extraction error message

if "wav_filename" not in st.session_state:
    st.session_state.wav_filename = None   # tracks which file was processed


# =============================================================================
# SECTION 3: Load Model Files
# =============================================================================

@st.cache_resource
def load_model_files():
    base          = os.path.dirname(os.path.abspath(__file__))
    model         = joblib.load(os.path.join(base, "models", "final_model.pkl"))
    scaler        = joblib.load(os.path.join(base, "models", "final_scaler.pkl"))
    feature_names = joblib.load(os.path.join(base, "models", "final_features.pkl"))
    explainer     = joblib.load(os.path.join(base, "models", "shap_explainer.pkl"))
    with open(os.path.join(base, "models", "final_model_name.txt"), "r") as f:
        model_name = f.read().strip()
    return model, scaler, feature_names, explainer, model_name

try:
    model, scaler, feature_names, explainer, model_name = load_model_files()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error    = str(e)


# =============================================================================
# SECTION 4: Show error if models failed to load
# =============================================================================

if not models_loaded:
    st.title("NeuroScreen AI — Setup Error")
    st.error("Could not load trained model files:")
    st.code(load_error)
    st.markdown("""
    **Run all pipeline scripts in order first:**
    ```
    python 2_preprocess.py
    python 3_train_models.py
    python 4_evaluate.py
    python 5_shap_explain.py
    python 6_save_model.py
    ```
    Then restart the app: `python -m streamlit run app.py`
    """)
    st.stop()


# =============================================================================
# SECTION 5: Title and Description
# =============================================================================

st.title("🧠 NeuroScreen AI")
st.subheader("Early Detection of Parkinson's Disease Using Voice Analysis")
st.markdown(f"""
This tool analyses **{len(feature_names)} voice measurements** to predict whether a patient
shows signs of Parkinson's disease.

> **Disclaimer:** This tool is for academic / research purposes only.
> It is **not** a substitute for professional medical diagnosis.
""")
st.markdown("---")


# =============================================================================
# SECTION 6: Sidebar
# =============================================================================

with st.sidebar:
    st.header("About This App")
    st.markdown(f"""
**Project:** NeuroScreen AI
**Disease:** Parkinson's Disease
**Dataset:** UCI Parkinson's Voice Dataset
**Model:** {model_name}
**Features:** {len(feature_names)} voice measurements
**Extractor:** Praat / Parselmouth
    """)

    st.markdown("---")
    st.header("How To Use")
    st.markdown("""
**Manual:** Enter the 16 feature values and click Predict.

**CSV:** Upload a CSV file with exactly 16 feature columns.

**Audio:** Upload a .wav file → features extracted automatically
→ click **Predict from Audio** to get the result.
    """)

    st.markdown("---")
    st.header("Recording Tips (WAV)")
    st.markdown("""
- Sustain the vowel **"aaah"** for **3-5 seconds**
- Record in a **quiet room**
- Hold the microphone **15-20 cm** from your mouth
- Use **mono, 22050 Hz** WAV format
- Minimum **2 seconds** of clear voice required
    """)

    st.markdown("---")
    st.header("Feature Reference")
    st.markdown("""
- **MDVP:Fo/Fhi/Flo** — Vocal frequency stats
- **Jitter variants** — Pitch period irregularity
- **Shimmer variants** — Amplitude irregularity
- **NHR / HNR** — Noise vs harmonic content
    """)


# =============================================================================
# SECTION 7: Input Method Selection
# =============================================================================

st.header("Step 1 — Provide Voice Input")

input_method = st.radio(
    "Choose how to provide voice data:",
    ["Enter values manually", "Upload a CSV file", "Upload a WAV audio file"],
    horizontal=True
)
st.markdown("---")


# =============================================================================
# SECTION 8: Default Values for Manual Entry (16 features only)
# Values from a real Parkinson's patient in the UCI dataset.
# =============================================================================

default_values = {
    "MDVP:Fo(Hz)"      : 119.992,
    "MDVP:Fhi(Hz)"     : 157.302,
    "MDVP:Flo(Hz)"     : 74.997,
    "MDVP:Jitter(%)"   : 0.00784,
    "MDVP:Jitter(Abs)" : 0.00007,
    "MDVP:RAP"         : 0.00370,
    "MDVP:PPQ"         : 0.00554,
    "Jitter:DDP"       : 0.01109,
    "MDVP:Shimmer"     : 0.04374,
    "MDVP:Shimmer(dB)" : 0.42600,
    "Shimmer:APQ3"     : 0.02182,
    "Shimmer:APQ5"     : 0.03130,
    "MDVP:APQ"         : 0.02971,
    "Shimmer:DDA"      : 0.06545,
    "NHR"              : 0.02211,
    "HNR"              : 21.033,
}

user_input      = {}
input_ready     = False   # True when we have valid data to predict on
source_is_audio = False   # True when input came from WAV upload


# =============================================================================
# SECTION 9A: Manual Input — 3-column grid of number fields
# =============================================================================

if input_method == "Enter values manually":

    st.subheader(f"Enter the {len(feature_names)} voice measurements:")
    st.caption("Default values shown are from a real Parkinson's patient in the UCI dataset.")

    col_groups = [feature_names[i:i+3] for i in range(0, len(feature_names), 3)]
    for group in col_groups:
        cols = st.columns(3)
        for col, feature in zip(cols, group):
            with col:
                user_input[feature] = st.number_input(
                    label=feature,
                    value=float(default_values.get(feature, 0.0)),
                    format="%.5f",
                    key=f"manual_{feature}"
                )

    input_ready = True   # manual form always has values


# =============================================================================
# SECTION 9B: CSV Upload
# =============================================================================

elif input_method == "Upload a CSV file":

    st.subheader("Upload a CSV file with one row of voice measurements:")
    st.caption(f"Your CSV must have exactly these {len(feature_names)} column headers (in any order):")
    st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            csv_df       = pd.read_csv(uploaded_file)
            missing_cols = [f for f in feature_names if f not in csv_df.columns]

            if missing_cols:
                st.error(f"Your CSV is missing these columns: {missing_cols}")
            else:
                for feat in feature_names:
                    user_input[feat] = float(csv_df[feat].iloc[0])
                st.success("CSV loaded successfully!")
                st.dataframe(csv_df[feature_names].head(1), use_container_width=True)
                input_ready = True

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("Please upload a CSV file to continue.")


# =============================================================================
# SECTION 9C: WAV Audio Upload — Parselmouth extraction
# =============================================================================

else:
    st.subheader("Upload a WAV voice recording:")

    st.info(
        "**Requirements:**  "
        "Say the vowel **'aaah'** clearly for **3-5 seconds** in a quiet room.  "
        "Save as a **.wav** file. The recording must be **at least 2 seconds** long "
        "and audible (not silent)."
    )

    wav_file = st.file_uploader("Choose a .wav audio file", type=["wav"])

    if wav_file is not None:

        # ----------------------------------------------------------------
        # Only re-extract if a NEW file was uploaded (avoid re-running
        # Praat on every Streamlit re-render / button click)
        # ----------------------------------------------------------------
        if wav_file.name != st.session_state.wav_filename:

            st.session_state.wav_filename = wav_file.name
            st.session_state.wav_features = None
            st.session_state.wav_error    = None

            with st.spinner("Extracting 16 voice features using Praat (Parselmouth)..."):

                # Save to temp file — Parselmouth needs a file path, not a buffer
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_file.getvalue())   # getvalue() reads without moving cursor
                    tmp_path = tmp.name

                features, error = extract_features(tmp_path)

                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            if error:
                st.session_state.wav_error    = error
                st.session_state.wav_features = None
            else:
                st.session_state.wav_features = features
                st.session_state.wav_error    = None

        # ----------------------------------------------------------------
        # Display playback widget (works even after re-run because
        # wav_file is still available from the uploader)
        # ----------------------------------------------------------------
        st.audio(wav_file, format="audio/wav")
        st.caption("Listen above to verify this is the correct recording.")

        # ----------------------------------------------------------------
        # Show extraction result from session state
        # ----------------------------------------------------------------
        if st.session_state.wav_error:
            st.error("Feature extraction failed:")
            st.warning(st.session_state.wav_error)
            st.markdown("""
            **Common fixes:**
            - Say **'aaah'** clearly and continuously for **at least 3 seconds**
            - Record in a **quiet room** (no fans, traffic, or background speech)
            - Make sure your microphone is working and the volume is not too low
            - Save as a proper **.wav** file (not .mp3 or .m4a renamed to .wav)
            """)

        elif st.session_state.wav_features is not None:

            st.success(f"✅ All {len(st.session_state.wav_features)} features extracted successfully!")

            # Show extracted values in two side-by-side tables
            st.markdown("#### Extracted Feature Values")
            st.caption(
                "These values were computed by Praat from your recording "
                "and will be passed to the model."
            )

            feat_df = pd.DataFrame(
                [(name, f"{val:.5f}") for name, val in st.session_state.wav_features.items()],
                columns=["Feature", "Extracted Value"]
            )

            half = len(feat_df) // 2
            col_left, col_right = st.columns(2)
            with col_left:
                st.dataframe(
                    feat_df.iloc[:half].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
            with col_right:
                st.dataframe(
                    feat_df.iloc[half:].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )

            # Load extracted features into user_input for prediction
            user_input      = st.session_state.wav_features
            source_is_audio = True
            input_ready     = True

    else:
        # New file upload — clear any previous extraction
        st.session_state.wav_features = None
        st.session_state.wav_error    = None
        st.session_state.wav_filename = None
        st.info("Please upload a .wav file to continue.")


# =============================================================================
# SECTION 10: Predict Button
# =============================================================================

st.markdown("---")
st.header("Step 2 — Get Prediction")

should_predict = False

if input_ready:
    btn_label = "🔍 Predict from Audio" if source_is_audio else "🔍 Predict"
    if st.button(btn_label, type="primary", use_container_width=True):
        should_predict = True
else:
    if input_method == "Enter values manually":
        st.info("Fill in all feature values above, then click Predict.")
    elif input_method == "Upload a CSV file":
        st.info("Upload a valid CSV file above to continue.")
    else:
        if st.session_state.wav_features is None and st.session_state.wav_error is None:
            st.info("Upload a WAV file above to continue.")
        elif st.session_state.wav_error:
            st.info("Fix the audio issues listed above, then upload a new file.")


# =============================================================================
# SECTION 11: Prediction Logic and Results
# =============================================================================

if should_predict:
    try:
        # Build input array in the exact feature order the model was trained on
        input_array   = np.array([[user_input[f] for f in feature_names]])
        input_scaled  = scaler.transform(input_array)

        prediction    = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence    = probabilities[prediction] * 100
        healthy_prob  = probabilities[0] * 100
        pk_prob       = probabilities[1] * 100

        # -----------------------------------------------------------------
        # Result
        # -----------------------------------------------------------------
        st.markdown("---")
        st.header("Result")

        if prediction == 1:
            st.error("⚠️ AT RISK — Signs of Parkinson's Disease Detected")
            result_label = "Parkinson's Disease"
        else:
            st.success("✅ HEALTHY — No Signs of Parkinson's Disease Detected")
            result_label = "Healthy"

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Prediction",  result_label)
        with c2: st.metric("Confidence",  f"{confidence:.1f}%")
        with c3: st.metric("Model Used",  model_name)

        if source_is_audio:
            st.caption("Prediction based on features automatically extracted from uploaded audio.")

        st.markdown("**Parkinson's Risk Probability:**")
        st.progress(int(pk_prob))
        st.caption(f"Healthy: {healthy_prob:.1f}%   |   Parkinson's: {pk_prob:.1f}%")

        # -----------------------------------------------------------------
        # SHAP Explanation Chart
        # -----------------------------------------------------------------
        st.markdown("---")
        st.header("Why did the model make this prediction?")
        st.caption(
            "**Red bars** = features pushing toward Parkinson's.   "
            "**Blue/Green bars** = features pushing toward Healthy."
        )

        try:
            shap_vals = np.array(explainer.shap_values(input_scaled))

            if shap_vals.ndim == 3:
                single_shap = shap_vals[1][0] if shap_vals.shape[0] == 2 \
                              else shap_vals[0, :, 1]
            elif shap_vals.ndim == 2:
                single_shap = shap_vals[0]
            else:
                single_shap = shap_vals.flatten()

            single_shap = [float(v) for v in single_shap]

            pairs     = sorted(zip(feature_names, single_shap),
                               key=lambda x: abs(x[1]), reverse=True)
            top_n     = min(len(feature_names), 16)
            top_feats = [p[0] for p in pairs[:top_n]][::-1]
            top_vals  = [p[1] for p in pairs[:top_n]][::-1]
            colors    = ["#FF4B4B" if v > 0 else "#21C55D" for v in top_vals]

            fig, ax = plt.subplots(figsize=(9, 6))
            bars = ax.barh(top_feats, top_vals, color=colors, edgecolor="white")
            for bar, val in zip(bars, top_vals):
                ax.text(
                    bar.get_width() + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8
                )
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value  (Red = toward Parkinson's, Green = toward Healthy)")
            ax.set_title(f"Feature Impact on This Prediction")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as shap_err:
            st.warning(f"SHAP chart could not be generated: {shap_err}")
            st.info("The prediction result above is still valid.")

        # -----------------------------------------------------------------
        # Input Summary Table (expandable)
        # -----------------------------------------------------------------
        st.markdown("---")
        label = (
            "View all 16 features extracted from audio"
            if source_is_audio else
            "View all 16 input values"
        )
        with st.expander(label):
            summary_df = pd.DataFrame(
                list(user_input.items()), columns=["Feature", "Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    except KeyError as ke:
        st.error(f"Feature name mismatch: {ke}")
        st.info(
            "The extracted feature names don't match the model's feature names. "
            "Re-run the full pipeline (2→6) and restart the app."
        )
    except Exception as pred_err:
        st.error(f"Prediction failed: {pred_err}")
        st.code(str(pred_err))


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(
    "NeuroScreen AI  |  B.E./B.Tech Final Year Project  |  "
    "UCI Parkinson's Voice Dataset  |  For academic use only."
)
