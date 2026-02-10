"""
Permian Methanol Injection Calculator - Streamlit App
Based on Permian Hydrate Curves data
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Hydrate curve data from "Permian Hyrdate Curves" tab
# Format: {methanol_rate_gal_per_mmscf: [(temp_F, pressure_psia), ...]}
HYDRATE_CURVES = {
    2: [
        (-14.18, 14.70), (8.48, 46.28), (19.49, 77.87), (26.94, 109.45), (32.59, 141.04),
        (37.17, 172.63), (41.01, 204.21), (43.41, 235.80), (45.13, 267.38), (46.66, 298.97),
        (48.02, 330.55), (49.25, 362.14), (50.37, 393.73), (51.39, 425.31), (52.33, 456.90),
        (53.20, 488.48), (54.01, 520.07), (54.76, 551.65), (55.46, 583.24), (56.11, 614.83),
        (56.72, 646.41), (57.30, 678.00), (57.85, 709.58), (58.36, 741.17), (58.84, 772.76),
        (59.30, 804.34), (59.74, 835.93), (60.15, 867.51), (60.55, 899.10), (60.92, 930.68),
        (61.28, 962.27), (61.62, 993.86), (61.95, 1025.44), (62.26, 1057.03), (62.56, 1088.61),
        (62.84, 1120.20), (63.11, 1151.79), (63.37, 1183.37), (63.62, 1214.96), (63.87, 1246.54),
        (64.10, 1278.13), (64.32, 1309.71), (64.53, 1341.30), (64.74, 1372.89), (64.94, 1404.47),
        (65.14, 1436.06), (65.34, 1467.64), (65.52, 1499.23), (65.70, 1530.81), (65.89, 1562.40),
    ],
    4: [
        (-12.67, 14.70), (10.13, 46.28), (21.22, 77.87), (28.73, 109.45), (33.43, 141.04),
        (35.98, 172.63), (38.11, 204.21), (39.94, 235.80), (41.53, 267.38), (42.95, 298.97),
        (44.21, 330.55), (45.35, 362.14), (46.40, 393.73), (47.35, 425.31), (48.24, 456.90),
        (49.05, 488.48), (49.81, 520.07), (50.52, 551.65), (51.19, 583.24), (51.81, 614.83),
        (52.40, 646.41), (52.96, 678.00), (53.48, 709.58), (53.98, 741.17), (54.45, 772.76),
        (54.90, 804.34), (55.32, 835.93), (55.73, 867.51), (56.12, 899.10), (56.49, 930.68),
        (56.84, 962.27), (57.18, 993.86), (57.50, 1025.44), (57.81, 1057.03), (58.11, 1088.61),
        (58.39, 1120.20), (58.67, 1151.79), (58.93, 1183.37), (59.18, 1214.96), (59.43, 1246.54),
        (59.67, 1278.13), (59.89, 1309.71), (60.11, 1341.30), (60.32, 1372.89), (60.53, 1404.47),
        (60.74, 1436.06), (60.95, 1467.64), (61.15, 1499.23), (61.35, 1530.81), (61.54, 1562.40),
    ],
    6: [
        (-11.27, 14.70), (11.67, 46.28), (22.82, 77.87), (27.28, 109.45), (30.15, 141.04),
        (32.20, 172.63), (34.09, 204.21), (35.72, 235.80), (37.14, 267.38), (38.41, 298.97),
        (39.56, 330.55), (40.60, 362.14), (41.55, 393.73), (42.44, 425.31), (43.26, 456.90),
        (44.03, 488.48), (44.74, 520.07), (45.42, 551.65), (46.06, 583.24), (46.66, 614.83),
        (47.23, 646.41), (47.77, 678.00), (48.28, 709.58), (48.77, 741.17), (49.24, 772.76),
        (49.68, 804.34), (50.11, 835.93), (50.52, 867.51), (50.91, 899.10), (51.28, 930.68),
        (51.64, 962.27), (51.98, 993.86), (52.32, 1025.44), (52.64, 1057.03), (52.94, 1088.61),
        (53.24, 1120.20), (53.52, 1151.79), (53.80, 1183.37), (54.07, 1214.96), (54.32, 1246.54),
        (54.57, 1278.13), (54.81, 1309.71), (55.05, 1341.30), (55.27, 1372.89), (55.49, 1404.47),
        (55.71, 1436.06), (55.93, 1467.64), (56.14, 1499.23), (56.35, 1530.81), (56.61, 1562.40),
    ],
    8: [
        (-9.97, 14.70), (13.10, 46.28), (20.23, 77.87), (23.40, 109.45), (25.77, 141.04),
        (27.67, 172.63), (29.27, 204.21), (30.67, 235.80), (31.93, 267.38), (32.52, 298.97),
        (33.52, 330.55), (34.45, 362.14), (35.32, 393.73), (36.14, 425.31), (36.91, 456.90),
        (37.64, 488.48), (38.34, 520.07), (39.00, 551.65), (39.63, 583.24), (40.24, 614.83),
        (40.81, 646.41), (41.36, 678.00), (41.89, 709.58), (42.40, 741.17), (42.88, 772.76),
        (43.35, 804.34), (43.80, 835.93), (44.23, 867.51), (44.64, 899.10), (45.04, 930.68),
        (45.42, 962.27), (45.79, 993.86), (46.15, 1025.44), (46.49, 1057.03), (46.82, 1088.61),
        (47.15, 1120.20), (47.46, 1151.79), (47.76, 1183.37), (48.05, 1214.96), (48.33, 1246.54),
        (48.61, 1278.13), (48.87, 1309.71), (49.13, 1341.30), (49.39, 1372.89), (49.63, 1404.47),
        (49.88, 1436.06), (50.12, 1467.64), (50.08, 1467.64), (50.67, 1530.81), (50.90, 1562.40),
    ],
    10: [
        (-8.75, 14.70), (12.42, 46.28), (16.02, 77.87), (18.12, 109.45), (19.59, 141.04),
        (20.77, 172.63), (21.83, 204.21), (22.84, 235.80), (23.84, 267.38), (24.81, 298.97),
        (25.78, 330.55), (26.72, 362.14), (27.64, 393.73), (28.54, 425.31), (29.41, 456.90),
        (30.24, 488.48), (31.05, 520.07), (31.83, 551.65), (32.44, 583.24), (32.47, 614.83),
        (32.66, 646.41), (33.28, 678.00), (33.87, 709.58), (34.44, 741.17), (34.99, 772.76),
        (35.52, 804.34), (36.03, 835.93), (36.52, 867.51), (37.00, 899.10), (37.45, 930.68),
        (37.89, 962.27), (38.31, 993.86), (38.73, 1025.44), (39.12, 1057.03), (39.50, 1088.61),
        (39.87, 1120.20), (40.23, 1151.79), (40.57, 1183.37), (40.91, 1214.96), (41.23, 1246.54),
        (41.54, 1278.13), (41.85, 1309.71), (42.16, 1341.30), (42.45, 1372.89), (42.74, 1404.47),
        (43.03, 1436.06), (43.31, 1467.64), (43.50, 1499.23), (43.88, 1530.81), (44.22, 1562.40),
    ],
    12: [
        (-7.60, 14.70), (8.55, 46.28), (9.66, 77.87), (9.10, 109.45), (8.31, 141.04),
        (8.20, 172.63), (8.83, 204.21), (9.87, 235.80), (11.10, 267.38), (12.40, 298.97),
        (13.69, 330.55), (14.96, 362.14), (16.18, 393.73), (17.35, 425.31), (18.47, 456.90),
        (19.54, 488.48), (20.57, 520.07), (21.54, 551.65), (22.47, 583.24), (23.36, 614.83),
        (24.21, 646.41), (25.03, 678.00), (25.81, 709.58), (26.57, 741.17), (27.28, 772.76),
        (27.97, 804.34), (28.64, 835.93), (29.28, 867.51), (29.90, 899.10), (30.50, 930.68),
        (31.07, 962.27), (31.63, 993.86), (32.17, 1025.44), (32.69, 1057.03), (32.81, 1088.61),
        (32.84, 1120.20), (32.86, 1151.79), (32.88, 1183.37), (32.91, 1214.96), (32.96, 1246.54),
        (33.33, 1278.13), (33.69, 1309.71), (34.04, 1341.30), (34.38, 1372.89), (34.72, 1404.47),
        (35.05, 1436.06), (35.37, 1467.64), (35.42, 1499.23), (35.92, 1530.81), (36.40, 1562.40),
    ],
    14: [
        (-6.52, 14.70), (2.49, 46.28), (-13.93, 77.87), (-15.93, 109.45), (-13.05, 141.04),
        (-10.09, 172.63), (-7.38, 204.21), (-4.93, 235.80), (-2.70, 267.38), (-0.68, 298.97),
        (1.18, 330.55), (2.89, 362.14), (4.48, 393.73), (5.95, 425.31), (7.33, 456.90),
        (8.63, 488.48), (9.84, 520.07), (10.99, 551.65), (12.07, 583.24), (13.10, 614.83),
        (14.08, 646.41), (15.00, 678.00), (15.89, 709.58), (16.74, 741.17), (17.55, 772.76),
        (18.32, 804.34), (19.07, 835.93), (19.78, 867.51), (20.47, 899.10), (21.13, 930.68),
        (21.77, 962.27), (22.38, 993.86), (22.97, 1025.44), (23.54, 1057.03), (24.10, 1088.61),
        (24.64, 1120.20), (25.16, 1151.79), (25.66, 1183.37), (26.16, 1214.96), (26.64, 1246.54),
        (27.11, 1278.13), (27.56, 1309.71), (28.01, 1341.30), (28.45, 1372.89), (28.89, 1404.47),
        (29.32, 1436.06), (29.74, 1467.64), (28.98, 1467.64), (29.70, 1499.23), (30.63, 1562.40),
    ],
    16: [
        (-5.50, 14.70), (-43.70, 46.28), (-35.02, 77.87), (-28.96, 109.45), (-24.29, 141.04),
        (-20.49, 172.63), (-17.27, 204.21), (-14.49, 235.80), (-12.04, 267.38), (-9.84, 298.97),
        (-7.86, 330.55), (-6.06, 362.14), (-4.40, 393.73), (-2.86, 425.31), (-1.43, 456.90),
        (-0.10, 488.48), (1.14, 520.07), (2.31, 551.65), (3.42, 583.24), (4.46, 614.83),
        (5.45, 646.41), (6.39, 678.00), (7.28, 709.58), (8.13, 741.17), (8.94, 772.76),
        (9.71, 804.34), (10.46, 835.93), (11.17, 867.51), (11.85, 899.10), (12.50, 930.68),
        (13.13, 962.27), (13.73, 993.86), (14.32, 1025.44), (14.88, 1057.03), (15.42, 1088.61),
        (15.95, 1120.20), (16.46, 1151.79), (16.94, 1183.37), (17.43, 1214.96), (17.89, 1246.54),
        (18.35, 1278.13), (18.79, 1309.71), (19.22, 1341.30), (19.65, 1372.89), (20.07, 1404.47),
        (20.49, 1436.06), (20.91, 1467.64), (20.67, 1499.23), (21.44, 1530.81), (22.19, 1562.40),
    ],
}

METHANOL_RATES = [2, 4, 6, 8, 10, 12, 14, 16]

# Colors for each curve
CURVE_COLORS = {
    2: '#1f77b4',   # blue
    4: '#ff7f0e',   # orange
    6: '#2ca02c',   # green
    8: '#d62728',   # red
    10: '#9467bd',  # purple
    12: '#8c564b',  # brown
    14: '#e377c2',  # pink
    16: '#7f7f7f',  # gray
}


def create_hydrate_chart(result, downstream_pressure):
    """
    Create a plotly chart showing all hydrate curves with the operating point marked.
    """
    fig = go.Figure()

    pressure_psia = downstream_pressure + 14.7
    t2 = result["t2"]

    # Plot each hydrate curve
    for rate in METHANOL_RATES:
        temps = [t for t, p in HYDRATE_CURVES[rate]]
        pressures = [p for t, p in HYDRATE_CURVES[rate]]

        fig.add_trace(go.Scatter(
            x=temps,
            y=pressures,
            mode='lines',
            name=f'{rate} gal/MMscf',
            line=dict(color=CURVE_COLORS[rate], width=2),
            hovertemplate=f'{rate} gal/MMscf<br>Temp: %{{x:.1f}}°F<br>Pressure: %{{y:.0f}} psia<extra></extra>'
        ))

    # Add horizontal line at operating pressure
    fig.add_hline(
        y=pressure_psia,
        line_dash="dash",
        line_color="black",
        line_width=1,
        annotation_text=f"Operating Pressure: {pressure_psia:.0f} psia",
        annotation_position="top right"
    )

    # Add vertical line at T2
    fig.add_vline(
        x=t2,
        line_dash="dash",
        line_color="black",
        line_width=1,
        annotation_text=f"T2: {t2:.1f}°F",
        annotation_position="top left"
    )

    # Add the operating point marker
    fig.add_trace(go.Scatter(
        x=[t2],
        y=[pressure_psia],
        mode='markers',
        name='Operating Point',
        marker=dict(
            size=15,
            color='red',
            symbol='x',
            line=dict(width=3, color='darkred')
        ),
        hovertemplate=f'Operating Point<br>T2: {t2:.1f}°F<br>Pressure: {pressure_psia:.0f} psia<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title='Permian Hydrate Formation Curves',
        xaxis_title='Temperature (°F)',
        yaxis_title='Pressure (psia)',
        xaxis=dict(range=[-20, 70]),
        yaxis=dict(range=[0, 1600]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest',
        height=500
    )

    return fig


def get_hydrate_temp_at_pressure(curve_data, pressure_psia):
    """
    Interpolate to find the hydrate formation temperature at a given pressure.
    """
    temps = [t for t, p in curve_data]
    pressures = [p for t, p in curve_data]

    if pressure_psia <= min(pressures):
        return temps[pressures.index(min(pressures))]
    if pressure_psia >= max(pressures):
        return temps[pressures.index(max(pressures))]

    # Linear interpolation
    for i in range(len(pressures) - 1):
        if pressures[i] <= pressure_psia <= pressures[i + 1]:
            ratio = (pressure_psia - pressures[i]) / (pressures[i + 1] - pressures[i])
            return temps[i] + ratio * (temps[i + 1] - temps[i])

    return temps[-1]


def calculate_methanol_rate(temperature, upstream_pressure, downstream_pressure, gas_rate_mmscf):
    """
    Calculate the required methanol injection rate using hydrate curve interpolation.
    """
    result = {
        "rate": None,
        "t2": None,
        "methanol_per_mmscf": None,
        "status": "success",
        "error": None,
        "hydrate_temps": {}
    }

    # Calculate adjusted temperature (T2) - accounts for Joule-Thomson cooling
    pressure_drop = upstream_pressure - downstream_pressure
    t2 = temperature - (pressure_drop / 100) * 8
    result["t2"] = t2

    # Convert downstream pressure from PSIG to PSIA (add atmospheric pressure)
    pressure_psia = downstream_pressure + 14.7

    # Check pressure bounds
    if pressure_psia > 1600:
        result["status"] = "error"
        result["error"] = f"Pressure {pressure_psia:.0f} psia exceeds curve data range (max ~1562 psia)"
        return result

    # Get hydrate formation temperature for each methanol rate at this pressure
    for rate in METHANOL_RATES:
        hydrate_temp = get_hydrate_temp_at_pressure(HYDRATE_CURVES[rate], pressure_psia)
        result["hydrate_temps"][rate] = hydrate_temp

    # Find the required methanol rate by interpolating between curves
    # If T2 is above the 2 gal/MMscf curve, minimal/no methanol needed
    # If T2 is below the 16 gal/MMscf curve, more than 16 gal/MMscf needed

    hydrate_temp_2 = result["hydrate_temps"][2]
    hydrate_temp_16 = result["hydrate_temps"][16]

    if t2 >= hydrate_temp_2:
        # Temperature is above hydrate formation even with minimal methanol
        result["rate"] = 0
        result["methanol_per_mmscf"] = 0
        result["status"] = "no_methanol"
        return result

    if t2 < hydrate_temp_16:
        # Temperature is below even the 16 gal/MMscf curve
        result["status"] = "error"
        result["error"] = f"T2 ({t2:.1f}°F) is below the 16 gal/MMscf hydrate curve ({hydrate_temp_16:.1f}°F). More methanol needed than curves support."
        return result

    # Interpolate between curves to find required rate
    methanol_per_mmscf = None
    for i in range(len(METHANOL_RATES) - 1):
        rate_low = METHANOL_RATES[i]
        rate_high = METHANOL_RATES[i + 1]
        temp_low = result["hydrate_temps"][rate_low]
        temp_high = result["hydrate_temps"][rate_high]

        if temp_high <= t2 <= temp_low:
            # Interpolate between these two rates
            ratio = (temp_low - t2) / (temp_low - temp_high)
            methanol_per_mmscf = rate_low + ratio * (rate_high - rate_low)
            break

    if methanol_per_mmscf is None:
        # Fallback - shouldn't happen but just in case
        methanol_per_mmscf = 16

    result["methanol_per_mmscf"] = methanol_per_mmscf

    # Calculate total daily rate
    methanol_rate = methanol_per_mmscf * gas_rate_mmscf

    # Round to nearest 0.25
    methanol_rate = round(methanol_rate * 4) / 4

    result["rate"] = methanol_rate

    return result


# Page configuration
st.set_page_config(
    page_title="Methanol Injection Calculator",
    page_icon="🧪",
    layout="wide"
)

# Title
st.title("Permian Methanol Injection Calculator")

# Create three columns: inputs on left, results in center, chart on right
left_col, center_col, right_col = st.columns([1, 1, 2])

with left_col:
    st.subheader("Inputs")

    gas_rate = st.number_input(
        "Gas Rate (MMscf/day)",
        min_value=0.0,
        max_value=1000.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        help="Highest expected gas rate"
    )

    temperature = st.number_input(
        "Lowest Upstream Temperature (°F)",
        min_value=-50.0,
        max_value=150.0,
        value=80.0,
        step=1.0,
        help="Lowest expected upstream temperature"
    )

    upstream_pressure = st.number_input(
        "Highest Upstream Pressure (PSIG)",
        min_value=0.0,
        max_value=5000.0,
        value=1100.0,
        step=25.0,
        help="Highest expected upstream pressure"
    )

    downstream_pressure = st.number_input(
        "Lowest Downstream Pressure (PSIG)",
        min_value=0.0,
        max_value=1500.0,
        value=500.0,
        step=25.0,
        help="Lowest expected downstream pressure"
    )

# Validate inputs and calculate
if upstream_pressure <= downstream_pressure:
    with center_col:
        st.subheader("Results")
        st.error("Upstream pressure must be greater than downstream pressure.")
elif gas_rate <= 0:
    with center_col:
        st.subheader("Results")
        st.error("Gas rate must be greater than zero.")
else:
    # Perform calculation
    result = calculate_methanol_rate(
        temperature, upstream_pressure, downstream_pressure, gas_rate
    )

    with center_col:
        st.subheader("Results")

        if result["status"] == "error":
            st.error(result["error"])
            if result["t2"] is not None:
                st.metric("Adjusted Temperature (T2)", f"{result['t2']:.1f} °F")

        elif result["status"] == "no_methanol":
            st.success("No methanol injection required!")
            st.metric("Methanol Rate", "0 gal/day")
            st.metric("Adjusted Temperature (T2)", f"{result['t2']:.1f} °F")

        else:
            st.metric("Required Methanol Rate", f"{result['rate']:.2f} gal/day")
            st.metric("Adjusted Temp (T2)", f"{result['t2']:.1f} °F")
            st.metric("Rate per MMscf", f"{result['methanol_per_mmscf']:.2f} gal/MMscf")

    with right_col:
        st.subheader("Hydrate Curves")
        fig = create_hydrate_chart(result, downstream_pressure)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red X = operating point")
