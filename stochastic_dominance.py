import numpy as np
from scipy.interpolate import interp1d

def fsd_dominance(x, y, grid=None):
    """
    First-order stochastic dominance: F_x(t) <= F_y(t) for all t, with strict for some.
    Returns True if x dominates y, False otherwise.
    """
    # Use empirical CDFs on a common grid
    if grid is None:
        all_vals = np.concatenate([x, y])
        grid = np.linspace(np.min(all_vals), np.max(all_vals), 100)
    # Compute ECDF
    def ecdf(data, t):
        return np.mean(data <= t)
    Fx = np.array([ecdf(x, t) for t in grid])
    Fy = np.array([ecdf(y, t) for t in grid])
    # Check Fx(t) <= Fy(t) for all t
    if np.all(Fx <= Fy + 1e-8) and np.any(Fx < Fy - 1e-8):
        return True
    return False

def ssd_dominance(x, y, grid=None):
    """
    Second-order stochastic dominance: integral of CDF (or equivalently, area under CDF).
    We use ∫ F_x(t) dt <= ∫ F_y(t) dt for all x, with strict.
    Alternatively, use quantile approach.
    """
    # Simple approach: sort and use piecewise integration
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    n = len(x)
    # Combine all points
    all_points = np.sort(np.concatenate([x_sorted, y_sorted]))
    # Compute cumulative distribution integrals
    def integral_cdf(data, points):
        # For each point, compute ∫_{-∞}^{point} F(t) dt ≈ sum over empirical
        # Simpler: use the area under the ECDF (trapezoidal)
        # We'll compute on the combined grid
        grid = points
        cdf_vals = np.array([np.mean(data <= t) for t in grid])
        # Integrate using trapezoidal rule
        integrals = np.cumsum(np.diff(grid) * (cdf_vals[:-1] + cdf_vals[1:]) / 2)
        return integrals
    Ix = integral_cdf(x, all_points)
    Iy = integral_cdf(y, all_points)
    if np.all(Ix <= Iy + 1e-8) and np.any(Ix < Iy - 1e-8):
        return True
    return False

def tsd_dominance(x, y, grid=None):
    """
    Third-order stochastic dominance: double integral of CDF.
    ∫∫ F_x(t) dt dt <= ∫∫ F_y(t) dt dt.
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_points = np.sort(np.concatenate([x_sorted, y_sorted]))
    def double_integral_cdf(data, points):
        # Compute first integral (area under CDF)
        cdf_vals = np.array([np.mean(data <= t) for t in points])
        # Trapezoidal integration
        int1 = np.cumsum(np.diff(points) * (cdf_vals[:-1] + cdf_vals[1:]) / 2)
        # Second integral: integrate the first integral
        # Need to align points: same points as before
        int1_full = np.insert(int1, 0, 0)  # length len(points)
        int2 = np.cumsum(np.diff(points) * (int1_full[:-1] + int1_full[1:]) / 2)
        return int2
    I2x = double_integral_cdf(x, all_points)
    I2y = double_integral_cdf(y, all_points)
    if np.all(I2x <= I2y + 1e-8) and np.any(I2x < I2y - 1e-8):
        return True
    return False

def dominance_score(returns, orders=[1,2,3], order_weights=None):
    """
    For a set of return vectors (list of arrays), compute for each asset the number of
    other assets it dominates (weighted by order).
    Returns: dict asset_idx -> score
    """
    n = len(returns)
    if order_weights is None:
        order_weights = {1:1, 2:1, 3:1}
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for order in orders:
                if order == 1 and fsd_dominance(returns[i], returns[j]):
                    scores[i] += order_weights.get(order, 1)
                    break   # if dominates at higher order, we count only once? But we can count all orders.
                elif order == 2 and ssd_dominance(returns[i], returns[j]):
                    scores[i] += order_weights.get(order, 1)
                    break
                elif order == 3 and tsd_dominance(returns[i], returns[j]):
                    scores[i] += order_weights.get(order, 1)
                    break
    return scores
