import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------------
# Constants for Stress Calculation and Sensor Colors
# -------------------------------------------------------
MODULUS = 3e9  # Young's modulus in Pascals (for stress calculation)
SENSOR_COLORS = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

# ==============================================================
# 0) Data Storage in Session State
# ==============================================================

if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = pd.DataFrame(columns=[
        "time_step", "sensor_name", "strain", "temperature", "health_index", "anomaly_flag"
    ])


def insert_sensor_reading(time_step, sensor_name, strain, temperature, health_index, anomaly_flag):
    new_row = {
        "time_step": time_step,
        "sensor_name": sensor_name,
        "strain": strain,
        "temperature": temperature,
        "health_index": health_index,
        "anomaly_flag": anomaly_flag
    }
    df = st.session_state.sensor_data
    df.loc[len(df)] = new_row
    st.session_state.sensor_data = df


def clear_sensor_data():
    st.session_state.sensor_data = pd.DataFrame(columns=[
        "time_step", "sensor_name", "strain", "temperature", "health_index", "anomaly_flag"
    ])


def get_all_data():
    return st.session_state.sensor_data.copy()


# ==============================================================
# 1) Multi-Sensor Creep Simulator (with Temperature-Dependent Creep)
# ==============================================================

class MultiSensorCreepSimulator:
    """
    Simulates sensor data for a rocket motor with creep (accumulated strain)
    that increases with temperature, plus cyclic and random variations.
    A "shock" event can be triggered at any time.
    """

    def __init__(self, n_sensors=2, creep_rate=1e-5, temp_baseline=25.0):
        self.n_sensors = n_sensors
        self.creep_rate = creep_rate
        self.temp_baseline = temp_baseline
        self.time_step = 0
        self.shock_active = False
        self.shock_counter = 0
        # Each sensor maintains its own accumulated strain offset.
        self.sensor_strain_offsets = [0.0] * n_sensors

    def generate_data(self, load_factor=1.0, temp_factor=1.0):
        data = []
        for i in range(self.n_sensors):
            # First, simulate a base temperature (without noise) to compute effective creep.
            temp_base = self.temp_baseline + 5.0 * temp_factor * math.sin(self.time_step * 0.05)
            # Temperature-dependent creep: e.g., 5% increase per degree above baseline.
            effective_creep_rate = self.creep_rate * (1 + 0.05 * (temp_base - self.temp_baseline))
            self.sensor_strain_offsets[i] += effective_creep_rate * load_factor

            strain = (self.sensor_strain_offsets[i] +
                      0.0005 * math.sin(self.time_step * 0.1) +
                      np.random.normal(0, 0.0001))
            # Now compute full temperature including noise.
            temperature = (self.temp_baseline +
                           5.0 * temp_factor * math.sin(self.time_step * 0.05) +
                           np.random.normal(0, 0.2))

            # Apply shock event if active.
            if self.shock_active:
                strain += 0.005  # Moderate spike
                temperature += 5.0

            data.append((i, strain, temperature))
        if self.shock_active:
            self.shock_counter -= 1
            if self.shock_counter <= 0:
                self.shock_active = False
        self.time_step += 1
        return data

    def trigger_shock_event(self, duration=5):
        self.shock_active = True
        self.shock_counter = duration


# ==============================================================
# 2) Health, Anomaly, Stress, and AI/ML Forecasting
# ==============================================================

def compute_health_index(strain, temperature):
    base_health = 100.0
    strain_penalty = abs(strain) * 1e4
    temp_penalty = max(0, temperature - 25) * 2
    health = base_health - strain_penalty - temp_penalty
    return max(0, min(100, health))


def detect_anomaly(strain, temperature, strain_threshold=0.008, temp_threshold=40.0):
    return 1 if abs(strain) > strain_threshold or temperature > temp_threshold else 0


def compute_stress(strain):
    MODULUS = 3e9  # Example Young's modulus in Pascals
    return strain * MODULUS / 1e6


def predict_remaining_life(df, critical_health=20):
    """
    Dummy AI/ML model:
      - Averages the health_index per time_step (only numeric column).
      - Uses linear regression to forecast remaining time steps until
        the average health drops below a critical threshold.
    (Replace this with an LSTM or Bayesian model for production use.)
    """
    if df.empty:
        return None
    # Group by time_step and average only the 'health_index'
    avg_df = df.groupby("time_step")["health_index"].mean().reset_index()
    if len(avg_df) < 2:
        return None
    m, b = np.polyfit(avg_df["time_step"], avg_df["health_index"], 1)
    if m >= 0:
        return None
    t_critical = (critical_health - b) / m
    remaining = t_critical - avg_df["time_step"].iloc[-1]
    return max(0, remaining)


# ==============================================================
# 3) 3D Cylinder Mesh for SRM Visualization
# ==============================================================

def generate_cylinder_mesh(radius=1.0, height=5.0, n_radial=20, n_height=10):
    thetas = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)
    zs = np.linspace(0, height, n_height)
    THETA, Z = np.meshgrid(thetas, zs)
    THETA_flat = THETA.ravel()
    Z_flat = Z.ravel()
    X_flat = radius * np.cos(THETA_flat)
    Y_flat = radius * np.sin(THETA_flat)
    faces = []
    for i in range(n_height - 1):
        for j in range(n_radial):
            v0 = i * n_radial + j
            v1 = i * n_radial + (j + 1) % n_radial
            v2 = (i + 1) * n_radial + j
            v3 = (i + 1) * n_radial + (j + 1) % n_radial
            faces.append((v0, v1, v2))
            faces.append((v1, v2, v3))
    faces = np.array(faces)
    return X_flat, Y_flat, Z_flat, faces


# ==============================================================
# 4) Custom Sensor Position Configuration
# ==============================================================

def configure_sensor_positions(n_sensors, radius, height):
    positions = []
    st.sidebar.subheader("Configure Sensor Positions")
    for i in range(n_sensors):
        st.sidebar.markdown(f"**Sensor {i + 1} Position**")
        default_x = radius * math.cos(2 * math.pi * i / n_sensors)
        default_y = radius * math.sin(2 * math.pi * i / n_sensors)
        default_z = 0.0
        x = st.sidebar.number_input(f"Sensor {i + 1} X", min_value=-radius, max_value=radius, value=default_x,
                                    key=f"s_{i}_x")
        y = st.sidebar.number_input(f"Sensor {i + 1} Y", min_value=-radius, max_value=radius, value=default_y,
                                    key=f"s_{i}_y")
        z = st.sidebar.number_input(f"Sensor {i + 1} Z", min_value=0.0, max_value=height, value=default_z,
                                    key=f"s_{i}_z")
        positions.append((x, y, z))
    return positions


def map_health_to_3d(X, Y, Z, sensor_positions, sensor_health):
    vertex_count = len(X)
    intensities = np.zeros(vertex_count)
    sensor_coords = np.array(sensor_positions)
    for i in range(vertex_count):
        vx, vy, vz = X[i], Y[i], Z[i]
        dists = np.sqrt((sensor_coords[:, 0] - vx) ** 2 +
                        (sensor_coords[:, 1] - vy) ** 2 +
                        (sensor_coords[:, 2] - vz) ** 2)
        closest_idx = np.argmin(dists)
        intensities[i] = sensor_health[closest_idx]
    return intensities


# ==============================================================
# 5) Streamlit App
# ==============================================================

def main():
    st.title("Digital Twin for Rocket Motor Health Monitoring")

    # Sidebar: Simulation & Display Controls
    st.sidebar.header("Simulation Parameters")
    n_sensors = st.sidebar.number_input("Number of Sensors", 1, 10, 3)
    load_factor = st.sidebar.slider("Load Factor", 0.5, 3.0, 1.0, 0.1)
    temp_factor = st.sidebar.slider("Temperature Factor", 0.5, 3.0, 1.0, 0.1)
    update_interval = st.sidebar.slider("Update Interval (s)", 0.1, 2.0, 0.5, 0.1)

    st.sidebar.header("Rocket Geometry")
    radius = st.sidebar.slider("Rocket Radius", 0.5, 2.0, 1.0, 0.1)
    height = st.sidebar.slider("Rocket Height", 2.0, 10.0, 5.0, 0.5)
    n_radial = st.sidebar.slider("Mesh Radial Segments", 8, 40, 20, 1)
    n_height = st.sidebar.slider("Mesh Height Segments", 5, 30, 10, 1)

    st.sidebar.header("Shock Event")
    shock_duration = st.sidebar.number_input("Shock Duration (steps)", 1, 20, 5)
    if st.sidebar.button("Trigger Shock"):
        if "sim" in st.session_state:
            st.session_state.sim.trigger_shock_event(shock_duration)
            st.sidebar.info("Shock triggered!")
        else:
            st.sidebar.warning("Simulator not initialized.")

    st.sidebar.header("Update Control")
    auto_update = st.sidebar.checkbox("Auto Update Simulation", value=True)

    # Configure custom sensor positions.
    sensor_positions = configure_sensor_positions(n_sensors, radius, height)

    # Display current simulation parameters for reference.
    st.sidebar.markdown("**Current Parameters:**")
    st.sidebar.write(f"Load Factor: {load_factor}")
    st.sidebar.write(f"Temperature Factor: {temp_factor}")

    # Initialize or reinitialize the simulator if sensor count changes.
    if "prev_n_sensors" not in st.session_state:
        st.session_state.prev_n_sensors = n_sensors
    if n_sensors != st.session_state.prev_n_sensors:
        # Do not clear historical data to preserve history.
        st.session_state.sim = MultiSensorCreepSimulator(n_sensors=n_sensors)
        st.session_state.prev_n_sensors = n_sensors
    if "sim" not in st.session_state:
        st.session_state.sim = MultiSensorCreepSimulator(n_sensors=n_sensors)
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False

    # Start/Pause Controls – visualizations remain when paused.
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.simulation_running:
            if st.button("Start Simulation"):
                st.session_state.simulation_running = True
        else:
            if st.button("Pause Simulation"):
                st.session_state.simulation_running = False

    # Auto-update if enabled and simulation is running.
    if st.session_state.simulation_running and auto_update:
        st_autorefresh(interval=update_interval * 1000, limit=1000, key="auto_refresh")
        update_simulation(load_factor, temp_factor)
    # If simulation is running but auto-update is off, allow manual stepping.
    if st.session_state.simulation_running and not auto_update:
        if st.button("Step Simulation"):
            update_simulation(load_factor, temp_factor)

    # Placeholders for visualizations.
    placeholder_2d_chart = st.empty()
    placeholder_3d = st.empty()
    placeholder_sensor_table = st.empty()
    placeholder_ml = st.empty()

    # Generate cylinder mesh for 3D visualization.
    X, Y, Z, faces = generate_cylinder_mesh(radius=radius, height=height, n_radial=n_radial, n_height=n_height)

    # Render dashboards.
    render_dashboard(placeholder_2d_chart, placeholder_3d, placeholder_sensor_table,
                     X, Y, Z, faces, sensor_positions, n_sensors, radius, height, placeholder_ml)


def update_simulation(load_factor, temp_factor):
    """Perform one simulation iteration and record sensor data."""
    batch_data = st.session_state.sim.generate_data(load_factor, temp_factor)
    time_step = st.session_state.sim.time_step
    for sensor_idx, strain, temperature in batch_data:
        hi = compute_health_index(strain, temperature)
        anomaly_flag = detect_anomaly(strain, temperature)
        insert_sensor_reading(time_step, f"Sensor_{sensor_idx + 1}", strain, temperature, hi, anomaly_flag)


def render_dashboard(placeholder_2d_chart, placeholder_3d, placeholder_sensor_table,
                     X, Y, Z, faces, sensor_positions, n_sensors, radius, height, placeholder_ml):
    df = get_all_data()
    if df.empty:
        placeholder_2d_chart.warning("No data yet. Click 'Start Simulation'.")
        placeholder_3d.write("No 3D visualization yet.")
        placeholder_sensor_table.write("No sensor data available yet.")
        placeholder_ml.write("No ML forecast available yet.")
        return

    # --- 1) 2D Health Chart (full history) ---
    sensor_groups = df.groupby("sensor_name")
    fig_2d = go.Figure()
    for sensor_name, group in sensor_groups:
        fig_2d.add_trace(go.Scatter(
            x=group["time_step"],
            y=group["health_index"],
            mode='lines+markers',
            name=sensor_name
        ))
    fig_2d.update_layout(
        title="Sensor Health Indices Over Time",
        xaxis_title="Time Step",
        yaxis_title="Health Index (0-100)",
        height=300
    )
    latest_step = int(df["time_step"].max())
    placeholder_2d_chart.plotly_chart(fig_2d, use_container_width=True, key=f"health_chart_{latest_step}")

    # --- 2) 3D Visualization ---
    latest_health = []
    for i in range(n_sensors):
        sname = f"Sensor_{i + 1}"
        sensor_df = df[df["sensor_name"] == sname]
        if not sensor_df.empty:
            latest_val = sensor_df["health_index"].iloc[-1]
        else:
            latest_val = 100.0
        latest_health.append(latest_val)
    intensities = map_health_to_3d(X, Y, Z, sensor_positions, latest_health)
    fig_3d = go.Figure(data=[
        go.Mesh3d(
            x=X, y=Y, z=Z,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=intensities,
            colorscale="RdYlGn",
            cmin=0, cmax=100,
            showscale=True,
            colorbar_title="Health Index",
            opacity=1.0
        )
    ])
    for i, pos in enumerate(sensor_positions):
        sensor_x, sensor_y, sensor_z = pos
        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        fig_3d.add_trace(go.Scatter3d(
            x=[sensor_x], y=[sensor_y], z=[sensor_z],
            mode="markers+text",
            text=[f"Sensor {i + 1}: {int(latest_health[i])}"],
            textposition="top center",
            marker=dict(
                size=8,
                color=color
            ),
            name=f"Sensor {i + 1}"
        ))
    fig_3d.update_layout(
        title="3D Rocket Motor Visualization (Color = Health Index)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=height / radius * 0.5)
        ),
        height=700
    )
    placeholder_3d.plotly_chart(fig_3d, use_container_width=True, key=f"3d_chart_{latest_step}")

    # --- 3) Sensor Data Table ---
    latest_data = []
    for sensor in sorted(df["sensor_name"].unique()):
        sensor_df = df[df["sensor_name"] == sensor]
        if not sensor_df.empty:
            latest_row = sensor_df.iloc[-1]
            strain = latest_row["strain"]
            temperature = latest_row["temperature"]
            hi = latest_row["health_index"]
            anomaly = latest_row["anomaly_flag"]
            stress = compute_stress(strain)
            latest_data.append({
                "Sensor": sensor,
                "Time Step": latest_row["time_step"],
                "Strain": round(strain, 6),
                "Stress (MPa)": round(stress, 2),
                "Temperature (°C)": round(temperature, 2),
                "Health Index": round(hi, 2),
                "Anomaly": anomaly
            })
    sensor_table_df = pd.DataFrame(latest_data)
    placeholder_sensor_table.subheader("Latest Sensor Values")
    placeholder_sensor_table.dataframe(sensor_table_df)

    # --- 4) AI/ML Forecast: Predict Remaining Life ---
    remaining = predict_remaining_life(df, critical_health=20)
    if remaining is not None:
        ml_text = f"**Predicted Remaining Life:** Approximately {remaining:.1f} time steps until average health reaches 20."
    else:
        ml_text = "Not enough data or health not declining to forecast remaining life."
    placeholder_ml.markdown(ml_text)


# ==============================================================
# 6) Future Improvement Ideas (Comments)
#
# - Replace the dummy linear forecast with an LSTM or Bayesian regression model
#   trained on historical sensor data.
# - Incorporate more realistic material properties, dynamic loading, and environmental
#   effects into the simulation. For example, use temperature-dependent creep laws and
#   FEA-generated stress/strain fields for interpolation onto the SRM surface.
# - Expand sensor modalities by including acoustic or vibration data.
# - Switch to a persistent database for long-term logging.
# ==============================================================

if __name__ == "__main__":
    main()
