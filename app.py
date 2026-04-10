import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.svar_model import SVAR
import plotly.graph_objects as go
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="GCC Macro Model", layout="wide")

# --- 2. THE SVAR ENGINE (Cached for speed) ---
@st.cache_data
def load_svar_model():
    """Loads historical data, builds the SVAR matrix, and extracts impact curves."""
    csv_data = """Date,Oil_Price,Strait_Capacity,KSA_GDP,UAE_GDP,QAT_GDP,KWT_GDP,OMN_GDP,BHR_GDP
    2019-03-31,63.1,100,1.7,2.0,1.5,1.2,2.1,1.8
    2019-06-30,68.9,95,0.5,1.1,0.2,0.0,2.3,1.0
    2019-09-30,62.0,90,-0.4,0.0,-1.5,-1.8,2.0,-0.5
    2019-12-31,63.2,100,0.3,1.5,1.0,0.8,2.5,1.2
    2020-03-31,50.4,100,-1.0,-0.5,-0.2,-0.5,1.0,-0.8
    2020-06-30,29.3,100,-7.0,-6.1,-3.5,-5.5,-3.0,-4.5
    2020-09-30,42.9,100,-4.6,-4.0,-2.0,-4.2,-1.5,-3.0
    2020-12-31,44.2,100,-3.9,-3.0,-1.1,-3.5,-1.0,-2.5
    2021-03-31,61.1,95,-0.1,0.5,-0.5,-1.0,1.5,0.0
    2021-06-30,68.9,95,1.8,2.2,1.0,1.5,2.8,1.2
    2021-09-30,73.4,90,7.0,5.5,2.1,3.0,4.5,2.5
    2021-12-31,79.7,90,6.7,5.0,2.5,3.5,4.2,2.8
    2022-03-31,97.9,100,9.9,7.5,4.8,6.0,5.5,4.0
    2022-06-30,113.9,100,12.2,8.8,6.1,7.5,6.0,5.5
    2022-09-30,97.7,100,8.8,6.5,5.0,6.2,5.2,4.5
    2022-12-31,88.5,100,5.5,4.2,3.8,4.5,4.0,3.5
    2023-03-31,81.2,95,3.8,3.5,2.0,2.5,3.5,2.0
    2023-06-30,78.0,95,1.2,2.0,0.5,1.0,3.0,1.2
    2023-09-30,86.7,90,-4.4,-2.5,-6.0,-5.5,1.5,-4.0
    2023-12-31,82.7,85,-3.7,-2.0,-7.5,-6.5,1.0,-5.5
    """
    df = pd.read_csv(io.StringIO(csv_data), index_col='Date', parse_dates=True)
    
    # 8x8 Matrix
    A_matrix = np.asarray([
        ['E', 0, 0, 0, 0, 0, 0, 0],
        ['E', 'E', 0, 0, 0, 0, 0, 0],
        ['E', 'E', 'E', 0, 0, 0, 0, 0],
        ['E', 'E', 0, 'E', 0, 0, 0, 0],
        ['E', 'E', 0, 0, 'E', 0, 0, 0],
        ['E', 'E', 0, 0, 0, 'E', 0, 0],
        ['E', 'E', 0, 0, 0, 0, 'E', 0],
        ['E', 'E', 0, 0, 0, 0, 0, 'E']
    ])
    
    model = SVAR(df, svar_type='A', A=A_matrix)
    results = model.fit(maxlags=1)
    irf = results.irf(periods=8) # Project 8 quarters (2 years)
    return irf.irfs

# Load the math engine
svar_curves = load_svar_model()

# --- 3. UI SIDEBAR CONTROLS ---
st.sidebar.header("Geopolitical Parameters")
st.sidebar.markdown("Adjust the sliders to simulate the 2026-2027 regional conflict impact.")

oil_price = st.sidebar.slider("Global Oil Price ($/bbl)", min_value=40, max_value=150, value=80, step=5)
strait_capacity = st.sidebar.slider("Strait of Hormuz Capacity (%)", min_value=0, max_value=100, value=100, step=5)
conflict_duration = st.sidebar.slider("Conflict Duration (Quarters)", min_value=0, max_value=8, value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Toggle Countries")
show_ksa = st.sidebar.checkbox("Saudi Arabia (KSA)", value=True)
show_uae = st.sidebar.checkbox("United Arab Emirates (UAE)", value=True)
show_qat = st.sidebar.checkbox("Qatar (QAT)", value=True)
show_kwt = st.sidebar.checkbox("Kuwait (KWT)", value=True)
show_omn = st.sidebar.checkbox("Oman (OMN)", value=True)
show_bhr = st.sidebar.checkbox("Bahrain (BHR)", value=True)

# --- 4. SCENARIO MATH LOGIC ---
# Calculate the severity of the shock based on the slider vs baseline (100%)
# A standard deviation in our dummy Strait data was ~5 points.
shock_magnitude = (100 - strait_capacity) / 5.0 
oil_bonus = (oil_price - 80) * 0.05

# Baselines for 2026
baselines = {'KSA': 4.5, 'UAE': 4.0, 'QAT': 2.5, 'KWT': 3.5, 'OMN': 2.6, 'BHR': 2.7}

# Map SVAR array indices
indices = {'KSA': 2, 'UAE': 3, 'QAT': 4, 'KWT': 5, 'OMN': 6, 'BHR': 7}
strait_index = 1

quarters = [f"Q{i+1}" for i in range(8)]
plot_data = pd.DataFrame({"Quarter": quarters})

# Apply SVAR calculations dynamically
for country, idx in indices.items():
    base_curve = np.full(8, baselines[country]) + oil_bonus
    svar_impact = svar_curves[:, idx, strait_index] * shock_magnitude
    
    # Apply shock only during the duration of the conflict
    for q in range(8):
        if q < conflict_duration:
            base_curve[q] -= svar_impact[q] 
        else:
            # Gradual recovery toward baseline
            base_curve[q] = base_curve[q-1] + ((baselines[country] - base_curve[q-1]) * 0.4)
            
    plot_data[country] = base_curve

# --- 5. DASHBOARD LAYOUT & PLOTTING ---
st.title("GCC Macroeconomic Scenario Dashboard")
st.markdown("Dynamic SVAR projections based on simulated maritime trade disruptions and energy shocks.")

# Summary Logic
if strait_capacity < 40 and conflict_duration > 2:
    st.error("🚨 **Systemic Shock Warning:** Severe recession territory for high-vulnerability states (Qatar, Kuwait, Bahrain).")
elif oil_price > 100 and strait_capacity > 80:
    st.success("📈 **Energy Boom Scenario:** Strong fiscal surpluses and elevated GDP growth across the bloc.")
else:
    st.info("⚖️ **Stable Baseline:** Adjust the parameters on the left to simulate a shock.")

# Build the interactive Plotly graph
fig = go.Figure()
colors = {'KSA': 'green', 'UAE': 'blue', 'QAT': 'maroon', 'KWT': 'orange', 'OMN': 'purple', 'BHR': 'red'}

for country in indices.keys():
    # Dynamically check the boolean variables from the sidebar (e.g., `show_ksa`)
    if locals()[f"show_{country.lower()}"]:
        line_dash = 'dash' if country in ['QAT', 'KWT', 'BHR'] else 'solid'
        fig.add_trace(go.Scatter(x=plot_data["Quarter"], y=plot_data[country], 
                                 mode='lines+markers', name=country,
                                 line=dict(color=colors[country], width=3, dash=line_dash)))

# Add zero-line and conflict zone highlighting
fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

if conflict_duration > 0:
    fig.add_vrect(x0=-0.5, x1=conflict_duration - 0.5, 
                  fillcolor="red", opacity=0.1, layer="below", line_width=0,
                  annotation_text="Active Conflict Phase", annotation_position="top left")

fig.update_layout(title="Projected Real GDP Growth (%) 2026-2027",
                  xaxis_title="Timeline (Quarters)",
                  yaxis_title="GDP Growth (%)",
                  height=600, hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)
