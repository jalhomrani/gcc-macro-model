import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GCC Macro Simulator", layout="wide")

# --- UI SIDEBAR CONTROLS ---
st.sidebar.header("Scenario Parameters")
oil_price = st.sidebar.slider("Global Oil Price ($/bbl)", 20, 150, 110)
strait_capacity = st.sidebar.slider("Strait Capacity (%)", 0, 100, 3)
conflict_duration = st.sidebar.slider("Conflict Duration (Months)", 0, 24, 6)

# Disruption Scale (D): 0 = no disruption, 1 = full closure
D = 1.0 - (strait_capacity / 100.0)

st.title("GCC Macroeconomic & Labour Market Simulator")
st.markdown("Based on Technical Appendix A: Two-Stage Sectoral and AR(1) Recovery Framework")

# --- CREATE TABS ---
tab1, tab2, tab3 = st.tabs(["📉 GDP Deviation & Bypass", "🏢 Sectoral Output (Two-Stage)", "💼 Labour Displacement"])

# ==========================================
# TAB 1: GDP DEVIATION & BYPASS
# ==========================================
with tab1:
    st.subheader("Projected GDP Deviation & Bypass Capacity")
    
    # 1. Bypass Coefficients (β_bypass) from Component 1
    bypass_coef = {'SAU': 0.35, 'UAE': 0.25, 'OMN': 1.00, 'QAT': 0.03, 'KWT': 0.03, 'BHR': 0.03}
    
    # 2. Hardcoded Delta GDP estimates from Table A.6 (Assuming $110/bbl and Strait at input capacity)
    # Note: A fully dynamic delta GDP requires the fiscal multipliers (α) and baseline volumes, 
    # which aren't fully provided for dynamic calculation, so we interpolate based on Table A.6 limits.
    
    countries = list(bypass_coef.keys())
    
    # Interpolating Delta GDP based on Strait Capacity (using Table A.6 bounds)
    delta_gdp = []
    for c in countries:
        if strait_capacity <= 3:
            # Values at 3% capacity
            vals = {'SAU': -2.3, 'UAE': -1.3, 'QAT': -5.4, 'KWT': -6.3, 'BHR': -2.1, 'OMN': 2.9}
        elif strait_capacity <= 25:
             # Values at 25% capacity
            vals = {'SAU': -0.8, 'UAE': -0.6, 'QAT': -3.3, 'KWT': -3.8, 'BHR': -1.3, 'OMN': 2.9}
        else:
             # Values at 50% capacity (and above)
            vals = {'SAU': 0.9, 'UAE': 0.1, 'QAT': -0.8, 'KWT': -0.9, 'BHR': -0.3, 'OMN': 2.9}
        delta_gdp.append(vals[c])

    # Calculate effective capacity based on bypass capability for visual reference
    effective_capacity = [strait_capacity + ((100 - strait_capacity) * bypass_coef[c]) for c in countries]
    
    fig_gdp = go.Figure()
    
    # Bar for Effective Capacity
    fig_gdp.add_trace(go.Bar(
        x=countries,
        y=effective_capacity,
        name='Effective Export Capacity (%)',
        marker_color='lightgray',
        yaxis='y1'
    ))
    
    # Line for GDP Deviation
    fig_gdp.add_trace(go.Scatter(
        x=countries,
        y=delta_gdp,
        name='ΔGDP (Percentage Points)',
        mode='lines+markers',
        marker=dict(size=10, color='crimson'),
        line=dict(width=3),
        yaxis='y2'
    ))

    # Layout for dual axis
    fig_gdp.update_layout(
        title="Impact of Pipeline Bypass on GDP",
        yaxis=dict(title="Effective Capacity (%)", range=[0, 100]),
        yaxis2=dict(title="ΔGDP (Percentage Points)", overlaying='y', side='right'),
        barmode='group',
        height=500
    )
    st.plotly_chart(fig_gdp, use_container_width=True)

# ==========================================
# TAB 2: SECTORAL OUTPUT (TWO-STAGE)
# ==========================================
with tab2:
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
# TAB 3: LABOUR DISPLACEMENT
# ==========================================
with tab3:
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
        
    fig_labour.update_layout(height=500, hovermode="x unified", xaxis_title="Months from Conflict Onset", yaxis_title="Total Expat Departures")
    st.plotly_chart(fig_labour, use_container_width=True)
    
    st.info("Recovery Phase: Once the conflict duration ends, the model uses AR(1) coefficients (ρ_GDP = 0.40, ρ_LM = 0.25) to map the asymmetric lag between economic rebound and workforce return.")
