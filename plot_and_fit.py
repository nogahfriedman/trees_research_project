"""
Plot cycle length data and fit to mathematical functions.

This script reads a CSV file with columns:
  - n: number of vertices
  - average_cycle_length (or cycle_length): average fundamental cycle length
  - std_dev (or std): standard deviation

It then:
1. Plots the data with error bars
2. Fits several candidate functions
3. Displays the best fit and residual statistics
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


# ──────────────────────────────────────────────
# Define candidate fitting functions
# ──────────────────────────────────────────────

def sqrt_linear(n, a, b):
    """Model: a * sqrt(n) + b"""
    return a * np.sqrt(n) + b


def linear(n, m, b):
    """Model: m * n + b"""
    return m * n + b


def sqrt_quadratic(n, a, b, c):
    """Model: a * sqrt(n) + b * n + c"""
    return a * np.sqrt(n) + b * n + c


def log_linear(n, a, b):
    """Model: a * log(n) + b"""
    return a * np.log(n) + b


def power_law(n, a, p):
    """Model: a * n^p"""
    return a * np.power(n, p)


# ──────────────────────────────────────────────
# Fitting and evaluation
# ──────────────────────────────────────────────

def fit_and_evaluate(n_data, y_data, y_err, func, initial_guess, func_name):
    """
    Fit data to function and compute residual statistics.
    
    Returns
    -------
    params : fitted parameters
    r_squared : R² value
    rmse : root mean squared error
    """
    try:
        params, _ = curve_fit(func, n_data, y_data, p0=initial_guess, sigma=y_err, absolute_sigma=True, maxfev=5000)
        y_pred = func(n_data, *params)
        
        # Compute R²
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))
        
        return params, r_squared, rmse
    except Exception as e:
        print(f"  Warning: Failed to fit {func_name}: {e}")
        return None, None, None


def _format_equation(name, params):
    """Format equation text based on model name and parameters."""
    if name.startswith("√n"):
        if len(params) == 2:
            return f"f(n) = {params[0]:+.6f}·√n  {params[1]:+.6f}"
        else:
            return f"f(n) = {params[0]:+.6f}·√n  {params[1]:+.6f}·n  {params[2]:+.6f}"
    elif name.startswith("Linear"):
        return f"f(n) = {params[0]:+.6f}·n  {params[1]:+.6f}"
    elif name.startswith("Log"):
        return f"f(n) = {params[0]:+.6f}·log(n)  {params[1]:+.6f}"
    elif name.startswith("Power"):
        return f"f(n) = {params[0]:+.6f}·n^{params[1]:.6f}"
    else:
        return str(params)


# ──────────────────────────────────────────────
# Main plotting and fitting
# ──────────────────────────────────────────────

def main(csv_file):
    """
    Load CSV, plot data, and fit to functions.
    """
    # Read CSV file
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Identify column names (handle variations)
    n_col = 'n'
    cycle_col = 'average_cycle_length' if 'average_cycle_length' in df.columns else 'cycle_length'
    std_col = 'std_dev' if 'std_dev' in df.columns else 'std'
    
    if cycle_col not in df.columns:
        print(f"Error: Could not find cycle length column. Available columns: {df.columns.tolist()}")
        return
    
    if std_col not in df.columns:
        print(f"Warning: Could not find std column. Using zeros.")
        y_err = np.zeros_like(df[cycle_col].values)
    else:
        y_err = df[std_col].values
    
    n_data = df[n_col].values
    y_data = df[cycle_col].values
    
    print(f"Loaded {len(n_data)} data points")
    print(f"n range: {n_data.min()} to {n_data.max()}")
    print(f"cycle_length range: {y_data.min():.4f} to {y_data.max():.4f}")
    print()
    
    # ──────────────────────────────────────────────
    # Fit different models
    # ──────────────────────────────────────────────
    
    models = [
        (sqrt_linear, [1.0, 0.0], "√n (a√n + b)"),
        (linear, [1.0, 0.0], "Linear (mn + b)"),
        (sqrt_quadratic, [1.0, 0.1, 0.0], "√n + Linear (a√n + bn + c)"),
        (log_linear, [1.0, 0.0], "Log (a·log(n) + b)"),
        (power_law, [1.0, 0.5], "Power law (a·n^p)"),
    ]
    
    print("Fitting models:")
    print("-" * 70)
    
    results = {}
    for func, initial_guess, name in models:
        params, r2, rmse = fit_and_evaluate(n_data, y_data, y_err, func, initial_guess, name)
        results[name] = (func, params, r2, rmse)
        
        if params is not None:
            print(f"{name:30s}  R²: {r2:8.6f}  RMSE: {rmse:10.6f}  Params: {params}")
        else:
            print(f"{name:30s}  Failed to fit")
    
    print()
    
    # Find best model
    valid_results = {k: v for k, v in results.items() if v[2] is not None}
    if not valid_results:
        print("Error: No models fitted successfully!")
        return
    
    best_name = max(valid_results.keys(), key=lambda k: valid_results[k][2])
    best_func, best_params, best_r2, best_rmse = valid_results[best_name]
    
    print(f"Best fit: {best_name}")
    print(f"  R² = {best_r2:.6f}")
    print(f"  RMSE = {best_rmse:.6f}")
    print(f"  Parameters: {best_params}")
    print()
    
    # ──────────────────────────────────────────────
    # Plot results
    # ──────────────────────────────────────────────
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data with best fit
    ax1.errorbar(n_data, y_data, yerr=y_err, fmt='o', markersize=6, capsize=4, label='Data', alpha=0.7)
    
    # Generate smooth curve for best fit
    n_smooth = np.linspace(n_data.min(), n_data.max(), 200)
    y_fit = best_func(n_smooth, *best_params)
    ax1.plot(n_smooth, y_fit, 'r-', linewidth=2, label=f'Best fit: {best_name}')
    
    # Build equation text for display
    equation_text = _format_equation(best_name, best_params)
    
    # Add text box with parameters and statistics
    textstr = f'{best_name}\n{equation_text}\n\nR² = {best_r2:.6f}\nRMSE = {best_rmse:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax1.set_xlabel('n (number of vertices)', fontsize=12)
    ax1.set_ylabel('Average cycle length', fontsize=12)
    ax1.set_title('Fundamental Cycle Length vs Graph Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    y_pred = best_func(n_data, *best_params)
    residuals = y_data - y_pred
    ax2.errorbar(n_data, residuals, yerr=y_err, fmt='o', markersize=6, capsize=4, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('n (number of vertices)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(csv_file).stem + "_fit.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    # ──────────────────────────────────────────────
    # Print fitted function equation
    # ──────────────────────────────────────────────
    
    print()
    print("Fitted function equations:")
    print("-" * 70)
    
    for name, (func, params, r2, rmse) in valid_results.items():
        if params is not None:
            if name.startswith("√n"):
                if len(params) == 2:
                    print(f"{name:30s}: f(n) = {params[0]:.6f}·√n + {params[1]:.6f}")
                else:
                    print(f"{name:30s}: f(n) = {params[0]:.6f}·√n + {params[1]:.6f}·n + {params[2]:.6f}")
            elif name.startswith("Linear"):
                print(f"{name:30s}: f(n) = {params[0]:.6f}·n + {params[1]:.6f}")
            elif name.startswith("Log"):
                print(f"{name:30s}: f(n) = {params[0]:.6f}·log(n) + {params[1]:.6f}")
            elif name.startswith("Power"):
                print(f"{name:30s}: f(n) = {params[0]:.6f}·n^{params[1]:.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find CSV files in current directory
        csv_files = list(Path(".").glob("*.csv"))
        if not csv_files:
            print("Usage: python plot_and_fit.py <csv_file>")
            print("\nNo CSV files found in current directory.")
            sys.exit(1)
        
        print(f"Found {len(csv_files)} CSV file(s):")
        for i, f in enumerate(csv_files):
            print(f"  {i}: {f}")
        
        csv_file = csv_files[0]
        print(f"\nUsing first file: {csv_file}")
    else:
        csv_file = sys.argv[1]
    
    main(csv_file)
