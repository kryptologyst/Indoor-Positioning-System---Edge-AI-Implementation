"""
Streamlit demo application for Indoor Positioning System.

This demo simulates edge constraints and provides real-time positioning visualization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.indoor_positioning import IndoorPositioningSystem
from utils.config import load_config


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'ips' not in st.session_state:
        st.session_state.ips = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'position_history' not in st.session_state:
        st.session_state.position_history = []
    if 'rssi_history' not in st.session_state:
        st.session_state.rssi_history = []
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []


def load_model():
    """Load the trained model."""
    try:
        config = load_config("configs/default.yaml")
        st.session_state.ips = IndoorPositioningSystem(config)
        
        # Generate and train a quick model for demo
        features, labels = st.session_state.ips.generate_synthetic_data()
        X_train, X_test, y_train, y_test = st.session_state.ips.prepare_data(features, labels)
        
        model = st.session_state.ips.build_model()
        st.session_state.ips.train_model(X_train, y_train, X_test, y_test)
        
        st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return False


def simulate_rssi_data(position: tuple, noise_level: float = 0.1) -> np.ndarray:
    """Simulate RSSI data for a given position."""
    x, y = position
    area_size = (20, 20)
    
    # Access point positions (corners)
    ap_positions = np.array([
        [0, 0],
        [area_size[0], 0],
        [0, area_size[1]],
        [area_size[0], area_size[1]]
    ])
    
    rssi_values = []
    for ap_x, ap_y in ap_positions:
        distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2)
        base_rssi = -30 - 2 * distance  # Simplified path loss
        noise = np.random.normal(0, noise_level * 10)
        rssi = np.clip(base_rssi + noise, -90, -30)
        rssi_values.append(rssi)
    
    return np.array(rssi_values)


def predict_position(rssi_data: np.ndarray) -> tuple:
    """Predict position from RSSI data."""
    if not st.session_state.model_loaded:
        return (0, 0)
    
    # Scale the RSSI data
    rssi_scaled = st.session_state.ips.scaler.transform(rssi_data.reshape(1, -1))
    
    # Predict position
    prediction = st.session_state.ips.model.predict(rssi_scaled, verbose=0)
    return tuple(prediction[0])


def create_position_plot():
    """Create the main position visualization plot."""
    if not st.session_state.position_history:
        return go.Figure()
    
    df = pd.DataFrame(st.session_state.position_history)
    
    fig = go.Figure()
    
    # Add actual positions
    fig.add_trace(go.Scatter(
        x=df['actual_x'],
        y=df['actual_y'],
        mode='markers',
        name='Actual Position',
        marker=dict(color='blue', size=8),
        text=df.index,
        hovertemplate='Actual: (%{x:.1f}, %{y:.1f})<br>Time: %{text}<extra></extra>'
    ))
    
    # Add predicted positions
    fig.add_trace(go.Scatter(
        x=df['predicted_x'],
        y=df['predicted_y'],
        mode='markers',
        name='Predicted Position',
        marker=dict(color='red', size=8),
        text=df.index,
        hovertemplate='Predicted: (%{x:.1f}, %{y:.1f})<br>Time: %{text}<extra></extra>'
    ))
    
    # Add access points
    ap_positions = [(0, 0), (20, 0), (0, 20), (20, 20)]
    for i, (ap_x, ap_y) in enumerate(ap_positions):
        fig.add_trace(go.Scatter(
            x=[ap_x],
            y=[ap_y],
            mode='markers',
            name=f'AP{i+1}',
            marker=dict(color='green', size=12, symbol='square'),
            showlegend=(i == 0)
        ))
    
    fig.update_layout(
        title="Indoor Positioning System - Real-time Tracking",
        xaxis_title="X Position (meters)",
        yaxis_title="Y Position (meters)",
        xaxis=dict(range=[-1, 21]),
        yaxis=dict(range=[-1, 21]),
        width=600,
        height=500
    )
    
    return fig


def create_rssi_plot():
    """Create RSSI signal strength plot."""
    if not st.session_state.rssi_history:
        return go.Figure()
    
    df = pd.DataFrame(st.session_state.rssi_history)
    
    fig = go.Figure()
    
    for i in range(4):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'ap{i+1}'],
            mode='lines+markers',
            name=f'AP{i+1}',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="RSSI Signal Strength Over Time",
        xaxis_title="Time Step",
        yaxis_title="RSSI (dBm)",
        yaxis=dict(range=[-90, -30]),
        width=600,
        height=300
    )
    
    return fig


def create_error_plot():
    """Create position error plot."""
    if not st.session_state.position_history:
        return go.Figure()
    
    df = pd.DataFrame(st.session_state.position_history)
    errors = np.sqrt((df['actual_x'] - df['predicted_x'])**2 + 
                    (df['actual_y'] - df['predicted_y'])**2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=errors,
        mode='lines+markers',
        name='Position Error',
        line=dict(color='red', width=2)
    ))
    
    # Add accuracy thresholds
    fig.add_hline(y=1, line_dash="dash", line_color="green", 
                  annotation_text="1m accuracy")
    fig.add_hline(y=2, line_dash="dash", line_color="orange", 
                  annotation_text="2m accuracy")
    fig.add_hline(y=5, line_dash="dash", line_color="red", 
                  annotation_text="5m accuracy")
    
    fig.update_layout(
        title="Position Error Over Time",
        xaxis_title="Time Step",
        yaxis_title="Error (meters)",
        width=600,
        height=300
    )
    
    return fig


def run_streamlit_app(config: dict, port: int = 8501):
    """Run the Streamlit demo application."""
    st.set_page_config(
        page_title="Indoor Positioning System Demo",
        page_icon="📍",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("📍 Indoor Positioning System Demo")
    st.markdown("**NOT FOR SAFETY-CRITICAL USE** - Research and educational purposes only")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Model loading
    if not st.session_state.model_loaded:
        if st.sidebar.button("Load Model", type="primary"):
            with st.spinner("Loading model..."):
                if load_model():
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
    
    # Simulation controls
    if st.session_state.model_loaded:
        st.sidebar.subheader("Simulation Settings")
        
        # Position controls
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x_pos = st.slider("X Position (m)", 0.0, 20.0, 10.0, 0.1)
        with col2:
            y_pos = st.slider("Y Position (m)", 0.0, 20.0, 10.0, 0.1)
        
        # Noise level
        noise_level = st.sidebar.slider("RSSI Noise Level", 0.0, 1.0, 0.1, 0.05)
        
        # Simulation buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📡 Scan Position"):
                simulate_position(x_pos, y_pos, noise_level)
        
        with col2:
            if st.button("🔄 Random Walk"):
                simulate_random_walk(noise_level)
        
        # Clear data
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.position_history = []
            st.session_state.rssi_history = []
            st.session_state.metrics_history = []
            st.rerun()
    
    # Main content
    if st.session_state.model_loaded:
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Scans", len(st.session_state.position_history))
        
        with col2:
            if st.session_state.position_history:
                recent_error = calculate_recent_error()
                st.metric("Recent Error", f"{recent_error:.2f}m")
            else:
                st.metric("Recent Error", "N/A")
        
        with col3:
            if st.session_state.position_history:
                accuracy_1m = calculate_accuracy_within_threshold(1.0)
                st.metric("1m Accuracy", f"{accuracy_1m:.1f}%")
            else:
                st.metric("1m Accuracy", "N/A")
        
        with col4:
            if st.session_state.position_history:
                accuracy_2m = calculate_accuracy_within_threshold(2.0)
                st.metric("2m Accuracy", f"{accuracy_2m:.1f}%")
            else:
                st.metric("2m Accuracy", "N/A")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_position_plot(), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_rssi_plot(), use_container_width=True)
        
        # Error plot
        st.plotly_chart(create_error_plot(), use_container_width=True)
        
        # Data table
        if st.session_state.position_history:
            st.subheader("Position History")
            df = pd.DataFrame(st.session_state.position_history)
            df['error'] = np.sqrt((df['actual_x'] - df['predicted_x'])**2 + 
                                 (df['actual_y'] - df['predicted_y'])**2)
            st.dataframe(df.round(2), use_container_width=True)
    
    else:
        st.info("👆 Please load the model using the sidebar controls to start the demo.")
        
        # Show system information
        st.subheader("System Information")
        st.markdown("""
        This demo simulates an Indoor Positioning System using WiFi RSSI signals.
        
        **Features:**
        - Real-time position estimation
        - RSSI signal visualization
        - Accuracy metrics tracking
        - Edge device simulation
        
        **How it works:**
        1. Load the trained neural network model
        2. Set a position using the sliders
        3. Click "Scan Position" to simulate RSSI measurement
        4. View the predicted vs actual position
        5. Monitor accuracy metrics over time
        """)


def simulate_position(x: float, y: float, noise_level: float):
    """Simulate a position scan."""
    # Generate RSSI data
    rssi_data = simulate_rssi_data((x, y), noise_level)
    
    # Predict position
    predicted_pos = predict_position(rssi_data)
    
    # Store data
    st.session_state.position_history.append({
        'actual_x': x,
        'actual_y': y,
        'predicted_x': predicted_pos[0],
        'predicted_y': predicted_pos[1],
        'timestamp': time.time()
    })
    
    st.session_state.rssi_history.append({
        'ap1': rssi_data[0],
        'ap2': rssi_data[1],
        'ap3': rssi_data[2],
        'ap4': rssi_data[3],
        'timestamp': time.time()
    })
    
    st.rerun()


def simulate_random_walk(noise_level: float):
    """Simulate a random walk pattern."""
    if not st.session_state.position_history:
        # Start from center
        x, y = 10.0, 10.0
    else:
        # Continue from last position
        last_pos = st.session_state.position_history[-1]
        x, y = last_pos['actual_x'], last_pos['actual_y']
        
        # Random step
        step_size = 2.0
        angle = np.random.uniform(0, 2 * np.pi)
        x += step_size * np.cos(angle)
        y += step_size * np.sin(angle)
        
        # Keep within bounds
        x = np.clip(x, 0, 20)
        y = np.clip(y, 0, 20)
    
    simulate_position(x, y, noise_level)


def calculate_recent_error() -> float:
    """Calculate recent position error."""
    if not st.session_state.position_history:
        return 0.0
    
    recent_positions = st.session_state.position_history[-5:]  # Last 5 positions
    errors = []
    
    for pos in recent_positions:
        error = np.sqrt((pos['actual_x'] - pos['predicted_x'])**2 + 
                       (pos['actual_y'] - pos['predicted_y'])**2)
        errors.append(error)
    
    return np.mean(errors)


def calculate_accuracy_within_threshold(threshold: float) -> float:
    """Calculate accuracy within given threshold."""
    if not st.session_state.position_history:
        return 0.0
    
    errors = []
    for pos in st.session_state.position_history:
        error = np.sqrt((pos['actual_x'] - pos['predicted_x'])**2 + 
                       (pos['actual_y'] - pos['predicted_y'])**2)
        errors.append(error)
    
    accuracy = np.mean(np.array(errors) <= threshold) * 100
    return accuracy


if __name__ == "__main__":
    # This will be called when running the demo
    config = load_config("configs/default.yaml")
    run_streamlit_app(config)
