import streamlit as st
import pandas as pd
import random
import plotly.express as px
from config import INPUT_FEATURES
from prediction import make_prediction_with_proba

st.set_page_config(
    page_title="Exoplanet Mission Control",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .stApp {
        background: url("https://images.pexels.com/photos/1169754/pexels-photo-1169754.jpeg");
        background-size: cover;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
    }

    .stButton>button {
        border: 2px solid #4A90E2;
        border-radius: 20px;
        color: #4A90E2;
        background-color: transparent;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #FFFFFF;
        color: #FFFFFF;
        background-color: #4A90E2;
    }
    
    .typing-text {
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        border-right: .15em solid orange;
        animation: typing 2s steps(30, end), blink-caret .75s step-end infinite;
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: orange; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Exoplanet Mission Control")
st.markdown("### Signal Analysis & Classification Terminal")

with st.sidebar:
    st.header("🚀 Mission Briefing")
    st.info(
        "Your objective is to analyze signals from Kepler Objects of Interest (KOIs) and classify their potential as exoplanets. "
        "Use the terminal below to process individual signals or entire datasets."
    )
    
    st.subheader("Signal Confidence Levels")
    st.success("**CONFIRMED:** > 90% Probability")
    st.warning("**CANDIDATE:** 40% - 90% Probability")
    st.error("**FALSE POSITIVE:** < 40% Probability")
    st.markdown("---")
    
    fun_facts = [
        "The first exoplanet was discovered in 1992, orbiting a pulsar.",
        "There are over 5,000 confirmed exoplanets, with thousands more candidates waiting for confirmation.",
        "Some exoplanets, known as 'hot Jupiters,' orbit their stars in just a few Earth days!",
        "The Kepler Space Telescope discovered most of the known exoplanets by watching for tiny dips in a star's brightness."
    ]
    st.subheader("🌠 Fun Fact")
    st.success(random.choice(fun_facts))
    st.markdown("---")

tab1, tab2 = st.tabs(["**Single Signal Analysis**", "**Bulk Data Processing**"])

with tab1:
    st.header("Manual Signal Input")

    FEATURE_LABELS = {
        'koi_fpflag_nt': ("Not Transit-Like Flag", "A flag indicating the signal is not transit-like."),
        'koi_fpflag_ss': ("Stellar Eclipse Flag", "A flag indicating the signal is from a stellar eclipse."),
        'koi_fpflag_co': ("Centroid Offset Flag", "A flag indicating a significant offset in the star's centroid."),
        'koi_fpflag_ec': ("Ephemeris Match Flag", "A flag indicating a match with a known celestial event."),
        'koi_period': ("Orbital Period [days]", "The time the KOI takes to complete one orbit around its star."),
        'koi_duration': ("Transit Duration [hours]", "The total duration of the observed transit event."),
        'koi_depth': ("Transit Depth [ppm]", "The amount of stellar light blocked, in parts per million."),
        'koi_impact': ("Impact Parameter", "The sky-projected distance between the star's center and the KOI's center during transit."),
        'koi_prad': ("Planetary Radius [Earth Radii]", "The radius of the KOI, measured in multiples of Earth's radius."),
        'koi_teq': ("Equilibrium Temperature [K]", "The estimated surface temperature of the KOI in Kelvin."),
        'koi_insol': ("Insolation Flux [Earth Flux]", "The amount of stellar radiation the KOI receives, relative to Earth."),
        'koi_steff': ("Stellar Temperature [K]", "The effective temperature of the host star in Kelvin."),
        'koi_slogg': ("Stellar Surface Gravity [log10(cm/s²)]", "The gravitational acceleration at the star's surface."),
        'koi_srad': ("Stellar Radius [Solar Radii]", "The radius of the host star, measured in multiples of the Sun's radius."),
        'koi_model_snr': ("Transit Signal-to-Noise Ratio", "The signal strength of the transit event compared to background noise.")
    }
    
    EXAMPLE_VALUES = {
        'koi_fpflag_nt': 0.0, 'koi_fpflag_ss': 0.0, 'koi_fpflag_co': 0.0,
        'koi_fpflag_ec': 0.0, 'koi_period': 83.2, 'koi_duration': 2.8,
        'koi_depth': 150.6, 'koi_impact': 0.15, 'koi_prad': 2.26,
        'koi_teq': 793.0, 'koi_insol': 93.5, 'koi_steff': 5455.0,
        'koi_slogg': 4.46, 'koi_srad': 0.92, 'koi_model_snr': 25.8
    }
    
    with st.container(border=True):
        cols = st.columns(3)
        input_dict = {}
        for i, feature in enumerate(INPUT_FEATURES):
            col = cols[i % 3]
            
            label_text, help_text = FEATURE_LABELS.get(feature, (feature.replace('_', ' ').title(), ""))
            example_value = EXAMPLE_VALUES.get(feature, 0.0)
            
            input_dict[feature] = col.number_input(
                label=label_text,
                help=help_text,
                value=float(example_value),
                format="%.6f",
                key=f"manual_{feature}"
            )

    if st.button("Initiate Scan 📡", use_container_width=True):
        input_df = pd.DataFrame([input_dict])
        
        with st.spinner("Scanning signal... Cross-referencing stellar database..."):
            result_df, probabilities = make_prediction_with_proba(input_df)
            prediction = result_df['prediction'].iloc[0]
            confidence = probabilities[0][result_df['prediction_label'].iloc[0]] * 100

            st.markdown("---")
            st.subheader("Signal Analysis Report")
            
            st.markdown(f'<div class="typing-text">{prediction}</div>', unsafe_allow_html=True)
            st.progress(int(confidence), text=f"Confidence Score: {confidence:.2f}%")
            
            if prediction == "CONFIRMED":
                st.success("Analysis complete. High-probability exoplanet signature detected. A momentous discovery!")
            elif prediction == "CANDIDATE":
                st.warning("Analysis complete. Signal shows promise. Flagged for further observation.")
            else:
                st.error("Analysis complete. Signal attributed to stellar noise or other non-planetary phenomena.")

with tab2:
    st.header("Bulk Data Feed")
    uploaded_file = st.file_uploader("Upload stellar data file (CSV)", type="csv", label_visibility="collapsed")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        st.subheader("Interactive Data Visualization")
        col1, col2, col3 = st.columns(3)
        x_axis = col1.selectbox("Select X-Axis", options=input_df.columns, index=0)
        y_axis = col2.selectbox("Select Y-Axis", options=input_df.columns, index=1)
        color_axis = col3.selectbox("Select Color Axis", options=input_df.columns)
        
        if x_axis and y_axis:
            fig = px.scatter(
                input_df, x=x_axis, y=y_axis, color=color_axis,
                title=f"{y_axis} vs. {x_axis}",
                template="plotly_dark",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        if st.button("Process Full Dataset 🛰️", use_container_width=True):
            with st.spinner("Executing batch analysis... This may take time."):
                result_df, _ = make_prediction_with_proba(input_df)
                st.subheader("Processing Complete: Results")
                st.dataframe(result_df)
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Processed Data", csv, "classified_data.csv", "text/csv", use_container_width=True)