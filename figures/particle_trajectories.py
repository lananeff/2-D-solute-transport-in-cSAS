import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation as animate
from scipy.integrate import quad
from velocities import u0, v0, u1, v1, upd, vpd, q, int_q, Q, uno, vno, us, vs, uL, vL
from src.d_functions import default_d
from scipy.integrate import solve_ivp
import numpy as np
import argparse
import os
from src.parameters import PARAMETERS
from src.figure_format import get_font_parameters, set_matplotlib_defaults
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

def plot_colored_trajectory(x, y, ax, cmap='viridis', label=None):
    """
    Plots a 2D trajectory with color indicating progression in time.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection with colors from a colormap
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.linspace(0, 1, len(x)))  # Set progression from light to dark
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    if label:
        ax.plot([], [], color=cm.get_cmap(cmap)(1.0), label=label)  # Legend proxy

    return line

# Get parameters for regime
def get_parameters(species, regime):
    """Retrieve parameters for the given species and regime."""
    try:
        species_params = PARAMETERS[species]
        if regime:
            regime_params = species_params.get(regime, {})
            params = {**species_params, **regime_params}  # Merge species and regime parameters
        else:
            params = species_params.copy()
        return params
    except KeyError:
        raise ValueError(f"Invalid species '{species}' or regime '{regime}'")

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Select CSF transport parameters.")
parser.add_argument("species", choices=["human", "mouse"], help="Choose species: human or mouse")
parser.add_argument("--regime", choices=["cardiac", "respiratory", "sleep"], help="Choose flow regime")

args = parser.parse_args()
params = get_parameters(args.species, args.regime)

print(f"Selected parameters for {args.species}, {args.regime if args.regime else 'all'}:")
for key, value in params.items():
    print(f"{key}: {value}")

# Set pars
A = params["A"]
eps = params["eps"]
alpha = params["alpha"]

plt.rcParams.update(get_font_parameters())
set_matplotlib_defaults()

z0 = [0.75, -0.9]
# Solve averaged system over τ = eps² * t, so t = τ / eps²
tmax = 20 * 2 * np.pi  # Full system runs on this
tau_span = (0, eps**2 * tmax)

def system_full(t, z, alpha, A, eps):
    x, y = z
    h = 1 + A * np.sin(t)
    h_prime = A * np.cos(t)
    dxdt = A * (u0(x, y, t, alpha, A, eps) + eps * u1(x, y, alpha, A, eps))
    dydt = (A / h) * (v0(x, y, t, alpha, A, eps) + eps * v1(x, y, alpha, A, eps)) - (A / h) * np.cos(t) * y
    return [dxdt, dydt]

# Averaged system (steady streaming)
def system_avg_L(t, z, alpha, A, eps):
    x, y = z
    dxdt = uL(x, y, alpha, A, eps)
    dydt = vL(x, y, alpha, A, eps)
    return [dxdt, dydt]

# Stokes drift only (oscillatory particle motion)
def system_stokes(t, z, alpha, A, eps):
    x, y = z
    h = 1 + A * np.sin(t)
    dxdt = A * u0(x, y, t, alpha, A, eps)
    dydt = (A/h) * v0(x, y, t, alpha, A, eps) - (A/h) * np.cos(t) * y
    return [dxdt, dydt]

def system_avg_drift(t, z, alpha, A, eps):
    x, y = z
    dxdt = us(x, y, alpha, A, eps)
    dydt = vs(x, y, alpha, A, eps)
    return [dxdt, dydt]



sol_full = solve_ivp(system_full, (0, tmax), z0, args=(alpha, A, eps), dense_output=True, method = 'RK45', max_step=0.1)
sol_avg = solve_ivp(system_avg_L, tau_span, z0, args=(alpha, A, eps), dense_output=True, method = 'RK45')
sol_stokes = solve_ivp(system_stokes, (0, tmax), z0, args=(alpha, A, eps), dense_output=True, method = 'RK45', max_step=0.1)
sol_drift = solve_ivp(system_avg_drift, tau_span, z0, args=(alpha, A, eps), dense_output=True, method = 'RK45')

# Evaluate solutions
t_full = np.linspace(0, tmax, 1000)
t_avg = np.linspace(*tau_span, 1000)

x_full, y_full = sol_full.sol(t_full)
x_avg, y_avg = sol_avg.sol(t_avg)
x_stokes, y_stokes = sol_stokes.sol(t_full)
x_drift, y_drift = sol_drift.sol(t_avg)

# Plot and save
os.makedirs("outputs", exist_ok=True)

# First plot: Full vs Averaged
fig, ax = plt.subplots()
plot_colored_trajectory(x_full, y_full, ax, cmap='plasma', label="Full")
ax.plot(x_avg, y_avg, label="Averaged (rescaled)", color='green')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("Comparison of Full and Averaged Trajectories")
os.makedirs("outputs/trajectorytest/old stokes", exist_ok=True)
plt.savefig(f"outputs/trajectorytest/old stokes/particle_trajectories_ic_{z0[0]:.2f}_{z0[1]:.2f}.png")
plt.close()

# Second plot: Stokes oscillatory vs drift
fig, ax = plt.subplots()
plot_colored_trajectory(x_stokes, y_stokes, ax, cmap='plasma', label="Oscillatory (Stokes)")
ax.plot(x_drift, y_drift, label="Stokes Drift", color='blue')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("Stokes Drift vs Oscillatory Motion")
plt.savefig(f"outputs/trajectorytest/old stokes/stokes_trajectories_ic_{z0[0]:.2f}_{z0[1]:.2f}.png")
plt.close()

print("Initial values of each trajectory:")
print(f"Full system:         x = {x_full[0]:.5f}, y = {y_full[0]:.5f}")
print(f"Averaged system:     x = {x_avg[0]:.5f}, y = {y_avg[0]:.5f}")
print(f"Stokes oscillatory:  x = {x_stokes[0]:.5f}, y = {y_stokes[0]:.5f}")
print(f"Stokes drift:        x = {x_drift[0]:.5f}, y = {y_drift[0]:.5f}")

print("Final values of each trajectory:")
print(f"Full system:         x = {x_full[-1]:.5f}, y = {y_full[-1]:.5f}")
print(f"Averaged system:     x = {x_avg[-1]:.5f}, y = {y_avg[-1]:.5f}")
print(f"Stokes oscillatory:  x = {x_stokes[-1]:.5f}, y = {y_stokes[-1]:.5f}")
print(f"Stokes drift:        x = {x_drift[-1]:.5f}, y = {y_drift[-1]:.5f}")