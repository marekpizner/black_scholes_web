import streamlit as st
import numpy as np
from option_pricing import (
    black_scholes_price,
    binomial_tree_price,
    generate_heatmap
)

# Page Configuration
st.set_page_config(
    page_title="Option Pricing Heatmaps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("📊 Option Pricing Heatmaps")
st.markdown("""
This application generates heatmaps for call and put option prices using various pricing models.
""")

# Sidebar Inputs
st.sidebar.header("Input Parameters")

S_spot_price = st.sidebar.number_input("Spot price", min_value=1.0, value=100.0, step=1.0)
S_steps = st.sidebar.slider("Number of Spot Price Steps", min_value=1.0, max_value=10.0, value=5.0)

vol_min = st.sidebar.number_input("Minimum Volatility (σ_min)", min_value=0.1, value=0.1, step=0.01)
vol_max = st.sidebar.number_input("Maximum Volatility (σ_max)", min_value=vol_min, value=0.3, step=0.01)
vol_steps = st.sidebar.slider("Number of Volatility Steps", min_value=10, max_value=50, value=20)

K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=0.01)
T = st.sidebar.number_input("Time to Maturity (T) in years", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
steps = st.sidebar.slider("Binomial Tree Steps", min_value=10, max_value=200, value=50, step=10)

# Generating Spot Price and Volatility Ranges
S_range = np.arange(S_spot_price - S_steps, S_spot_price + S_steps + 1)
vol_range = np.linspace(vol_min, vol_max, vol_steps)

# Define models to plot
models = {
    "Black-Scholes": black_scholes_price,
    "Binomial Tree": binomial_tree_price,
}

model_params = {
    "binomial_tree_price": {"steps": steps}
}

# Generate Heatmaps and Display Prices for Each Model
for model_name, model_func in models.items():
    st.subheader(f"{model_name} Model")
    
    # Calculate price for a specific example: S=100, sigma=0.2
    example_S = 100
    example_sigma = 0.2
    example_params = model_params.get(model_func.__name__, {})
    
    try:
        call_price = model_func(example_S, K, T, r, example_sigma, **example_params, option_type="call")
        put_price = model_func(example_S, K, T, r, example_sigma, **example_params, option_type="put")
    except TypeError:
        # Handle models that do not require all parameters
        call_price = model_func(example_S, K, T, r, example_sigma, option_type="call")
        put_price = model_func(example_S, K, T, r, example_sigma, option_type="put")

    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"Call Option Price: {call_price:.2f}")  # Display in red
        call_heatmap_fig = generate_heatmap(S_range, vol_range, K, T, r, model_func, example_params, option_type="call")
        st.pyplot(call_heatmap_fig)
    
    with col2:
        st.error(f"Put Option Price: {put_price:.2f}")  # Display in green
        put_heatmap_fig = generate_heatmap(S_range, vol_range, K, T, r, model_func, example_params, option_type="put")
        st.pyplot(put_heatmap_fig)


# Footer
st.markdown("---")
st.markdown("Created by [Marek Pizner](https://www.linkedin.com/in/marek-pizner-33a7a5150/)")
