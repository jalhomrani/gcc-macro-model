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
    return results.irf(periods=8).irfs

svar_curves = load_svar_model()

# --- 3. UI SIDEBAR CONTROLS ---
st.sidebar.header("Geopolitical Parameters")
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

# Shared Variables
quarters = [f"Q{i+1}" for i in range(8)]
colors = {'KSA': 'green', 'UAE': 'blue', 'QAT': 'maroon', 'KWT': 'orange', 'OMN': 'purple', 'BHR': 'red'}
indices = {'KSA': 2, 'UAE': 3, 'QAT': 4, 'KWT': 5, 'OMN': 6, 'BHR': 7}

st.title("GCC Macroeconomic Scenario Dashboard")
if strait_capacity < 40 and conflict_duration > 2:
    st.error("🚨 **Systemic Shock Warning:** Severe recession, high inflation, and labor flight detected.")
else:
    st.info("Adjust the parameters on the left to simulate a shock.")

# --- CREATE TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📉 Real GDP Growth", "🛒 Cost-Push Inflation", "💼 Labor Market", "🏢 Sectoral Impact"])

# ==========================================
# TAB 1: GDP SIMULATION
# ==========================================
with tab1:
    st.subheader("Projected Real GDP Growth (%) 2026-2027")
    shock_magnitude = (100 - strait_capacity) / 5.0 
    oil_bonus = (oil_price - 80) * 0.05
    baselines_gdp = {'KSA': 4.5, 'UAE': 4.0, 'QAT': 2.5, 'KWT': 3.5, 'OMN': 2.6, 'BHR': 2.7}
    
    gdp_data = pd.DataFrame({"Quarter": quarters})
    for country, idx in indices.items():
        base_curve = np.full(8, baselines_gdp[country]) + oil_bonus
        svar_impact = svar_curves[:, idx, 1] * shock_magnitude
        
        for q in range(8):
            if q < conflict_duration:
                base_curve[q] -= svar_impact[q] 
            else:
                base_curve[q] = base_curve[q-1] + ((baselines_gdp[country] - base_curve[q-1]) * 0.4)
        gdp_data[country] = base_curve

    fig_gdp = go.Figure()
    for country in indices.keys():
        if locals()[f"show_{country.lower()}"]:
            line_dash = 'dash' if country in ['QAT', 'KWT', 'BHR'] else 'solid'
            fig_gdp.add_trace(go.Scatter(x=gdp_data["Quarter"], y=gdp_data[country], mode='lines+markers', name=country, line=dict(color=colors[country], width=3, dash=line_dash)))

    fig_gdp.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    if conflict_duration > 0:
        fig_gdp.add_vrect(x0=-0.5, x1=conflict_duration - 0.5, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Active Conflict Phase", annotation_position="top left")
    
    fig_gdp.update_layout(height=500, hovermode="x unified", yaxis_title="GDP Growth (%)")
    st.plotly_chart(fig_gdp, use_container_width=True)

# ==========================================
# TAB 2: INFLATION SIMULATION
# ==========================================
with tab2:
    st.subheader("Projected Consumer Price Index (CPI) Inflation (%)")
    base_inflation = 2.0
    inf_vuln = {'KSA': 0.8, 'UAE': 0.8, 'QAT': 2.5, 'KWT': 2.2, 'OMN': 0.2, 'BHR': 2.0}
    inf_data = pd.DataFrame({"Quarter": quarters})
    strait_penalty = (100 - strait_capacity) * 0.15 
    
    for country in indices.keys():
        inf_curve = np.full(8, base_inflation)
        country_shock = strait_penalty * inf_vuln[country]
        for q in range(8):
            if q < conflict_duration:
                inf_curve[q] = base_inflation + country_shock + (q * 0.5)
            else:
                inf_curve[q] = inf_curve[q-1] - ((inf_curve[q-1] - base_inflation) * 0.3)
        inf_data[country] = inf_curve

    fig_inf = go.Figure()
    for country in indices.keys():
        if locals()[f"show_{country.lower()}"]:
            line_dash = 'dash' if country in ['QAT', 'KWT', 'BHR'] else 'solid'
            fig_inf.add_trace(go.Scatter(x=inf_data["Quarter"], y=inf_data[country], mode='lines+markers', name=country, line=dict(color=colors[country], width=3, dash=line_dash)))

    fig_inf.add_hline(y=2.0, line_dash="dot", line_color="green", line_width=2, annotation_text="Target Rate")
    if conflict_duration > 0:
        fig_inf.add_vrect(x0=-0.5, x1=conflict_duration - 0.5, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Active Conflict Phase", annotation_position="top left")
    
    fig_inf.update_layout(height=500, hovermode="x unified", yaxis_title="Inflation (%)")
    st.plotly_chart(fig_inf, use_container_width=True)

# ==========================================
# TAB 3: LABOR MARKET SIMULATION
# ==========================================
with tab3:
    st.subheader("Expatriate Workforce Growth/Contraction (%)")
    st.markdown("Models the 'flight of talent' caused by stalled projects, high inflation, and regional instability. Negative values indicate mass expatriate departures.")
    
    # Baseline expat population growth (normally highly positive as economies expand)
    base_expat_growth = {'KSA': 3.0, 'UAE': 4.5, 'QAT': 1.5, 'KWT': 1.0, 'OMN': 2.0, 'BHR': 1.5}
    
    # Vulnerability: How easily can the expat workforce leave? 
    # UAE and KSA have deeper roots ("Golden Visas"), while KWT and BHR see faster cyclical turnover.
    flight_vuln = {'KSA': 0.5, 'UAE': 0.4, 'QAT': 1.2, 'KWT': 1.5, 'OMN': 0.3, 'BHR': 1.4}
    
    labor_data = pd.DataFrame({"Quarter": quarters})
    conflict_stress = (100 - strait_capacity) * 0.08  # The stress factor driving expats away
    
    for country in indices.keys():
        labor_curve = np.full(8, base_expat_growth[country])
        shock = conflict_stress * flight_vuln[country]
        
        for q in range(8):
            if q < conflict_duration:
                # Expat flight gets worse the longer the conflict drags on (compounding)
                labor_curve[q] = base_expat_growth[country] - shock - (q * 1.2)
            else:
                # Labor markets take a long time to trust the region again; slow recovery
                labor_curve[q] = labor_curve[q-1] + ((base_expat_growth[country] - labor_curve[q-1]) * 0.25)
                
        labor_data[country] = labor_curve

    fig_labor = go.Figure()
    for country in indices.keys():
        if locals()[f"show_{country.lower()}"]:
            line_dash = 'dash' if country in ['QAT', 'KWT', 'BHR'] else 'solid'
            fig_labor.add_trace(go.Scatter(x=labor_data["Quarter"], y=labor_data[country], mode='lines+markers', name=country, line=dict(color=colors[country], width=3, dash=line_dash)))

    # Zero line separates growth (hiring) from contraction (layoffs/flight)
    fig_labor.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, annotation_text="Net Zero Migration")
    
    if conflict_duration > 0:
        fig_labor.add_vrect(x0=-0.5, x1=conflict_duration - 0.5, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Active Conflict Phase", annotation_position="top left")
    
    fig_labor.update_layout(height=500, hovermode="x unified", yaxis_title="Expat Population Growth (%)")
    st.plotly_chart(fig_labor, use_container_width=True)

# ==========================================
# TAB 4: SECTORAL IMPACT SIMULATION
# ==========================================
with tab4:
    st.subheader("Economic Sector Performance Projection (%)")
    st.markdown("Highlights the divergence between 'Safe Haven/Defense' sectors and 'Vulnerable/Consumer' sectors during a crisis.")
    
    # Baseline growth rates for key GCC sectors in a non-conflict scenario
    sectors = ['Energy & Petrochem', 'Tourism & Aviation', 'Defense & Cyber', 'Real Estate', 'Logistics']
    baselines_sec = [3.0, 6.0, 4.0, 4.5, 3.5]
    
    # Calculate dynamic impacts based on the sidebar sliders
    # Strait penalty hurts consumer/trade, helps defense
    strait_penalty = 100 - strait_capacity
    
    # Mathematical logic for sectoral divergence
    energy_proj = 3.0 + ((oil_price - 80) * 0.05) - (strait_penalty * 0.1)
    tourism_proj = 6.0 - (strait_penalty * 0.05) - (conflict_duration * 0.8)
    defense_proj = 4.0 + (strait_penalty * 0.04) + (conflict_duration * 0.5)
    real_estate_proj = 4.5 - (strait_penalty * 0.03) - (conflict_duration * 0.4)
    logistics_proj = 3.5 - (strait_penalty * 0.04)
    
    projected_sec = [energy_proj, tourism_proj, defense_proj, real_estate_proj, logistics_proj]
    
    # Create the DataFrame
    sector_df = pd.DataFrame({
        'Sector': sectors,
        'Baseline Growth (%)': baselines_sec,
        'Projected Growth (%)': projected_sec
    })
    
    # Build the Bar Chart
    fig_sec = go.Figure()
    
    fig_sec.add_trace(go.Bar(
        x=sector_df['Sector'],
        y=sector_df['Baseline Growth (%)'],
        name='Pre-War Baseline',
        marker_color='lightgray'
    ))
    
    # Dynamically color the projected bars (Green for growth, Red for contraction)
    proj_colors = ['red' if val < 0 else 'royalblue' for val in sector_df['Projected Growth (%)']]
    
    fig_sec.add_trace(go.Bar(
        x=sector_df['Sector'],
        y=sector_df['Projected Growth (%)'],
        name='Conflict Projection',
        marker_color=proj_colors
    ))
    
    fig_sec.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
    fig_sec.update_layout(barmode='group', height=500, yaxis_title="Growth Rate (%)")
    
    st.plotly_chart(fig_sec, use_container_width=True)
    
    # Dynamic Summary Analysis
    worst_sector = sector_df.loc[sector_df['Projected Growth (%)'].idxmin()]
    best_sector = sector_df.loc[sector_df['Projected Growth (%)'].idxmax()]
    
    st.info(f"**Insight:** Capital is flowing out of **{worst_sector['Sector']}** (projected at {worst_sector['Projected Growth (%)']:.1f}%) and pivoting toward **{best_sector['Sector']}** (projected at {best_sector['Projected Growth (%)']:.1f}%).")
