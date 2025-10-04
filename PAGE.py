import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# --- Streamlit setup ---
st.set_page_config(page_title="Microgravity Pharma Hub", layout="wide")

st.title("Microgravity Pharmaceutical Manufacturing Hub")
st.markdown("""
Simulate and optimize protein crystal manufacturing in Low Earth Orbit (LEO).  
All sections dynamically respond to temperature, pH, radiation, and concentration changes.
""")

# --- Train AI model ---
@st.cache_resource
def train_ai_model():
    np.random.seed(42)
    temps = np.random.uniform(10, 60, 400)
    concs = np.random.uniform(0.01, 0.1, 400)
    phs = np.random.uniform(6.5, 7.5, 400)
    radiation = np.random.uniform(0, 0.1, 400)
    X = np.column_stack((temps, concs, phs, radiation))
    y = (0.6 * (30 - abs(temps - 30)) / 30 +
         0.5 * (0.05 - abs(concs - 0.05)) / 0.05 +
         0.4 * (7 - abs(phs - 7)) / 0.5 -
         0.2 * radiation +
         np.random.normal(0, 0.03, 400))
    y = np.clip(y, 0, 1)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

model = train_ai_model()

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
temp = st.sidebar.slider("Temperature (°C)", 10.0, 60.0, 30.0)
conc = st.sidebar.slider("Concentration (g/L)", 0.01, 0.1, 0.05)
ph = st.sidebar.slider("pH Level", 6.5, 7.5, 7.0)
radiation = st.sidebar.slider("Radiation Exposure (mGy/day)", 0.0, 0.1, 0.05)
batch_size = st.sidebar.number_input("Batch Size (kg)", 0.1, 100.0, 10.0)
optimize = st.sidebar.button("Optimize with AI")

# --- Simulation ---
def simulate_crystal_growth(temp, conc, ph, radiation, gravity):
    input_data = np.array([[temp, conc, ph, radiation]])
    ai_quality = model.predict(input_data)[0]
    gravity_factor = np.exp(-gravity * 1e5) * 0.1  
    radiation_penalty = radiation * 0.2
    quality = np.clip(ai_quality + gravity_factor - radiation_penalty, 0, 1)
    yield_pct = quality * 100
    return yield_pct, quality

micro_yield, micro_quality = simulate_crystal_growth(temp, conc, ph, radiation, gravity=0.000001)
earth_yield, earth_quality = simulate_crystal_growth(temp, conc, ph, radiation, gravity=9.81)

# --- Section 1: Simulation Results ---
st.header("Section 1: Simulation Results")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Yield and Quality")
    improvement = micro_yield - earth_yield
    st.metric("Microgravity Yield (%)", f"{micro_yield:.2f}", f"{improvement:.2f}")
    st.metric("Earth Yield (%)", f"{earth_yield:.2f}")
    st.metric("Microgravity Quality", f"{micro_quality:.3f}")
    st.metric("Earth Quality", f"{earth_quality:.3f}")

with col2:
    st.subheader("Crystal Growth Over Time")
    t = np.linspace(0, 24, 100)
    micro_growth = micro_yield * (1 - np.exp(-t / (6 + radiation * 20)))
    earth_growth = earth_yield * (1 - np.exp(-t / (10 + radiation * 50)))
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=t, y=micro_growth, mode="lines", name="Microgravity", line=dict(color="#00cc96", width=3)))
    fig_growth.add_trace(go.Scatter(x=t, y=earth_growth, mode="lines", name="Earth", line=dict(color="#ff7f0e", width=3)))
    fig_growth.update_layout(
        template="plotly_dark",
        xaxis_title="Time (hours)",
        yaxis_title="Cumulative Yield (%)",
        title="Crystal Growth Curve"
    )
    st.plotly_chart(fig_growth, use_container_width=True)

# --- Section 2: Economic Analysis ---
st.header("Section 2: Economic Analysis")
launch_cost = 2_000_000
maintenance_cost = 500_000
material_cost_per_kg = 100_000
operational_cost = launch_cost + maintenance_cost + material_cost_per_kg * batch_size

yield_factor = (micro_yield / (earth_yield + 1e-6))
efficiency_gain = np.clip((yield_factor - 1) * 100, -100, 300)
savings = operational_cost * (efficiency_gain / 100) * 0.4
reduced_cost = operational_cost - savings

# Display cost table
cost_df = pd.DataFrame({
    "Component": ["Launch", "Maintenance", "Materials", "Total Cost", "Savings", "Reduced Cost"],
    "Amount ($)": [launch_cost, maintenance_cost, material_cost_per_kg * batch_size,
                   operational_cost, int(savings), int(reduced_cost)]
})
st.table(cost_df)

# Cost per kg curve
batch_sizes = np.linspace(0.1, 100, 50)
total_costs = launch_cost + maintenance_cost + material_cost_per_kg * batch_sizes
costs_per_kg = total_costs / batch_sizes / (yield_factor if yield_factor > 0 else 1)

fig_cost = go.Figure()
fig_cost.add_trace(go.Scatter(
    x=batch_sizes,
    y=costs_per_kg,
    mode="lines+markers",
    line=dict(color="#1f77b4", width=3),
    name="Cost per kg"
))

# --- Auto zoom-in ---
ymin = np.min(costs_per_kg) * 0.95
ymax = np.min(costs_per_kg) * 1.35

fig_cost.update_layout(
    title="Cost per kg vs. Batch Size",
    xaxis_title="Batch Size (kg)",
    yaxis_title="Cost per kg ($)",
    yaxis=dict(range=[ymin, ymax]),
    template="plotly_dark"
)
st.plotly_chart(fig_cost, use_container_width=True)

# --- Section 3: Sustainability Metrics ---
st.header("Section 3: Sustainability Metrics")
energy_efficiency = 0.8 + micro_quality * 0.2
debris_score = 0.1 * (1 - micro_quality)
longevity = 8 + micro_quality * 5
st.write(f"**Energy Efficiency:** {(energy_efficiency * 100):.1f}% (Solar optimized)")
st.write(f"**Debris Impact Score:** {debris_score:.2f} (Lower is better)")
st.write(f"**Platform Longevity:** {longevity:.1f} years (Based on quality stability)")
st.progress(min(energy_efficiency, 1.0))

# --- Section 4: 3D Crystal Visualization ---
st.header("Section 4: 3D Crystal Structure Visualization")
lattice_density = micro_quality * 8 + 4
spacing = 1 / (micro_quality + 0.1)
x, y, z = np.mgrid[0:lattice_density:spacing, 0:lattice_density:spacing, 0:lattice_density:spacing]
x, y, z = x.flatten(), y.flatten(), z.flatten()
color_intensity = np.sin(x + y + z) * micro_quality

fig_3d = go.Figure(data=go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=micro_quality * 6 + 2,
        color=color_intensity,
        colorscale='Viridis',
        opacity=0.8
    )
))
fig_3d.update_layout(
    template="plotly_dark",
    title=f"Protein Crystal Lattice Visualization (Quality: {micro_quality:.2f})",
    scene=dict(
        xaxis_title="X (Å)",
        yaxis_title="Y (Å)",
        zaxis_title="Z (Å)",
        aspectmode='cube'
    )
)
st.plotly_chart(fig_3d, use_container_width=True)