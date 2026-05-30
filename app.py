import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GCC Macro Simulator", layout="wide")

# --- UI SIDEBAR CONTROLS ---
st.sidebar.header("Scenario Parameters")
oil_price = st.sidebar.slider("Global Oil Price ($/bbl)", 20, 150, 72)
strait_capacity = st.sidebar.slider("Strait Capacity (%)", 0, 100, 85)
conflict_duration = st.sidebar.slider("Conflict Duration (Months)", 0, 24, 6)

# Disruption Scale (D): 0 = normal, 1 = total closure
D = 1.0 - (strait_capacity / 100.0)

st.title("GCC Macroeconomic & Labour Market Simulator")
st.markdown("Based on Technical Appendix A: Two-Stage Sectoral and AR(1) Recovery Framework")

# --- CREATE TABS ---
tab1, tab2, tab3 = st.tabs(["🏢 Sectoral Output (Two-Stage)", "💼 Labour Displacement", "📉 GDP Bypass & Recovery"])

# ==========================================
# TAB 1: SECTORAL OUTPUT (TWO-STAGE)
# ==========================================
with tab1:
    st.subheader("Sector Output Projections (100 = Pre-Conflict Baseline)")
    
    # Coefficients from Table A.3
    sectors = ['Tourism & Aviation', 'Transport & Storage', 'Construction', 
               'Manufacturing', 'Wholesale & Retail', 'Financial Services', 
               'Energy & Utilities', 'Government Services']
    
    mu_1 = [-25, -8, 2, 1, 2, 3, 0, 0]
    mu_2 = [-15, -12, -8, -5, -4, -3, 0, -6]
    
    months = np.arange(0, 25)
    sector_df = pd.DataFrame({"Month": months})
    
    for i, s in enumerate(sectors):
        output_index = []
        for t in months:
            if t <= 6:
                # Stage 1 (0 to 6 months)
                val = 100 + (mu_1[i] * D * (t / 6.0))
            else:
                # Stage 2 (6 to 24 months)
                stage1_end = 100 + (mu_1[i] * D)
                val = stage1_end + (mu_2[i] * D * (min(t - 6, 18) / 18.0))
            output_index.append(val)
        sector_df[s] = output_index

    fig_sec = go.Figure()
    for s in sectors:
        line_dash = 'dash' if s in ['Construction', 'Government Services', 'Manufacturing'] else 'solid'
        fig_sec.add_trace(go.Scatter(x=sector_df['Month'], y=sector_df[s], mode='lines', name=s, line=dict(width=3, dash=line_dash)))

    fig_sec.add_vline(x=6, line_dash="dot", line_color="red", annotation_text="Stage 2: Fiscal Channel Activates")
    fig_sec.add_hline(y=100, line_color="black", line_width=1)
    
    if conflict_duration > 0:
        fig_sec.add_vrect(x0=0, x1=conflict_duration, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Conflict Active")
        
    fig_sec.update_layout(height=550, hovermode="x unified", xaxis_title="Months from Conflict Onset", yaxis_title="Output Index")
    st.plotly_chart(fig_sec, use_container_width=True)

# ==========================================
# TAB 2: LABOUR DISPLACEMENT
# ==========================================
with tab2:
    st.subheader("Cumulative Non-National Workforce Departures")
    
    L0 = 27970273  # GCC Base non-national workforce
    lambda_rate = -0.025  # Quarterly departure rate
    
    labour_df = pd.DataFrame({"Month": months})
    base_disp, low_disp, high_disp = [], [], []
    
    for t in months:
        # L(t) = L0 * (1 + lambda * D)^(t/3)
        l_base = L0 * ((1 + (lambda_rate * D)) ** (t / 3.0))
        l_low = L0 * ((1 + (lambda_rate * 0.70 * D)) ** (t / 3.0))
        l_high = L0 * ((1 + (lambda_rate * 1.30 * D)) ** (t / 3.0))
        
        base_disp.append(L0 - l_base)
        low_disp.append(L0 - l_low)
        high_disp.append(L0 - l_high)
        
    fig_labour = go.Figure()
    
    # Uncertainty Bounds
    fig_labour.add_trace(go.Scatter(x=labour_df['Month'], y=high_disp, mode='lines', line=dict(width=0), showlegend=False))
    fig_labour.add_trace(go.Scatter(x=labour_df['Month'], y=low_disp, mode='lines', fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', line=dict(width=0), name='±30% Uncertainty Range'))
    
    # Core Projection
    fig_labour.add_trace(go.Scatter(x=labour_df['Month'], y=base_disp, mode='lines', line=dict(color='navy', width=4), name='Projected Departures'))
    
    if conflict_duration > 0:
        fig_labour.add_vrect(x0=0, x1=conflict_duration, fillcolor="red", opacity=0.1, layer="below", line_width=0)
        
    fig_labour.update_layout(height=500, hovermode="x unified", xaxis_title="Months", yaxis_title="Total Expat Departures")
    st.plotly_chart(fig_labour, use_container_width=True)

# ==========================================
# TAB 3: GDP BYPASS & RECOVERY
# ==========================================
with tab3:
    st.subheader("Infrastructure Bypass Capability (Effective Strait Access)")
    
    bypass_coef = {'OMN': 1.00, 'SAU': 0.35, 'UAE': 0.25, 'QAT': 0.03, 'KWT': 0.03, 'BHR': 0.03}
    
    # Calculate effective capacity based on bypass capability
    countries = list(bypass_coef.keys())
    effective_capacity = [strait_capacity + ((100 - strait_capacity) * bypass_coef[c]) for c in countries]
    
    fig_bypass = go.Bar(x=countries, y=effective_capacity, marker_color=['purple', 'green', 'blue', 'maroon', 'orange', 'red'])
    
    layout = go.Layout(height=400, yaxis_title="Effective Export Capacity (%)", 
                       title="How Pipelines & Coastal Terminals Mitigate Strait Closure")
    st.plotly_chart(go.Figure(data=fig_bypass, layout=layout), use_container_width=True)
    
    st.info("Recovery Phase: Once the conflict duration ends, the model uses AR(1) coefficients (ρ_GDP = 0.40, ρ_LM = 0.25) to map the asymmetric lag between economic rebound and workforce return.")
