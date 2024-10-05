import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Calculate Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price

def binomial_tree_price(S, K, T, r, sigma, steps, option_type="call"):
    """Calculate option price using a binomial tree."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    
    prices = np.zeros(steps + 1)
    prices[0] = S * d ** steps
    for i in range(1, steps + 1):
        prices[i] = prices[i - 1] * u / d
    
    values = np.maximum(0, (prices - K) if option_type == "call" else (K - prices))
    
    for _ in range(steps):
        values[:-1] = np.exp(-r * dt) * (q * values[1:] + (1 - q) * values[:-1])
    
    return values[0]


def generate_heatmap(S_range, vol_range, K, T, r, model_func, model_params={}, steps=50, option_type="call"):
    """Generate a heatmap of option prices using the specified model."""
    prices = np.zeros((len(vol_range), len(S_range)))
    
    for i, sigma in enumerate(vol_range):
        for j, S in enumerate(S_range):
            if model_func.__name__ == "binomial_tree_price":
                prices[i, j] = model_func(S, K, T, r, sigma, steps, option_type)
            else:
                prices[i, j] = model_func(S, K, T, r, sigma, option_type)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(prices, xticklabels=round_values(S_range), yticklabels=round_values(vol_range),
                annot=False, fmt=".2f", cmap="coolwarm", ax=ax)
    
    x_ticks = np.linspace(0, len(S_range) - 1, min(5, len(S_range))).astype(int)
    y_ticks = np.linspace(0, len(vol_range) - 1, min(5, len(vol_range))).astype(int)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(round_values(S_range[x_ticks]), rotation=45)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(round_values(vol_range[y_ticks]))
    
    ax.set_title(f'{model_func.__name__.replace("_", " ").capitalize()} - {option_type.capitalize()}')
    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel('Volatility (σ)')
    
    plt.tight_layout()
    return fig

def round_values(arr):
    """Helper function to round values for ticks."""
    return np.round(arr, 2)
