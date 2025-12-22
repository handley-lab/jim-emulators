#!/usr/bin/env python
"""
Parameter sensitivity analysis for IMRPhenomXPHM waveforms.

Generates colorbar plots showing how waveform properties (amplitude, phase,
real, imaginary parts) vary as individual parameters are changed.

Usage:
    python scripts/parameter_sensitivity.py [--output-dir figures/]
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import sys

sys.path.insert(0, "src")

from jim_emulators.waveforms.lal_waveforms import (
    WaveformParameters,
    generate_fd_waveform,
    get_waveform_amplitude_phase,
)
from jim_emulators.waveforms.utils import (
    physical_to_geometric_frequency,
    fd_to_td,
    geometric_fd_to_td,
    interpolate_to_uniform_grid,
)


# Fiducial parameter sets for exploration
FIDUCIAL_PARAMS = {
    "equal_mass_nonspinning": {
        "mass_1": 30.0,
        "mass_2": 30.0,
        "chi1z": 0.0,
        "chi2z": 0.0,
        "luminosity_distance": 100.0,
        "inclination": 0.0,
        "phase": 0.0,
    },
    "unequal_mass_aligned_spin": {
        "mass_1": 35.0,
        "mass_2": 25.0,
        "chi1z": 0.3,
        "chi2z": -0.2,
        "luminosity_distance": 100.0,
        "inclination": np.pi / 4,
        "phase": 0.0,
    },
    "high_mass_high_spin": {
        "mass_1": 60.0,
        "mass_2": 40.0,
        "chi1z": 0.8,
        "chi2z": 0.6,
        "luminosity_distance": 100.0,
        "inclination": np.pi / 3,
        "phase": 0.0,
    },
}

# Parameters to vary and their ranges (min, max, n_points, label)
N_GRID_POINTS = 30  # Number of parameter values to sample

PARAMETER_RANGES = {
    "mass_1": (20.0, 80.0, N_GRID_POINTS, "Primary Mass $m_1$ [$M_\\odot$]"),
    "mass_2": (10.0, 50.0, N_GRID_POINTS, "Secondary Mass $m_2$ [$M_\\odot$]"),
    "chi1z": (-0.99, 0.99, N_GRID_POINTS, "Primary Spin $\\chi_{1z}$"),
    "chi2z": (-0.99, 0.99, N_GRID_POINTS, "Secondary Spin $\\chi_{2z}$"),
    "inclination": (0.0, np.pi, N_GRID_POINTS, "Inclination $\\iota$ [rad]"),
    "symmetric_mass_ratio": (0.1, 0.25, N_GRID_POINTS, "Symmetric Mass Ratio $\\eta$"),
}


def masses_from_total_and_eta(M_total, eta):
    """Convert total mass and symmetric mass ratio to component masses."""
    m1 = M_total * (1 + np.sqrt(1 - 4 * eta)) / 2
    m2 = M_total - m1
    return m1, m2


def generate_waveform_grid(
    fiducial: dict,
    param_name: str,
    param_values: np.ndarray,
    f_min: float = 10.0,
    f_max: float = 512.0,
    delta_f: float = 0.125,
) -> tuple:
    """
    Generate a grid of waveforms varying one parameter.

    Returns
    -------
    frequencies : np.ndarray
        Common frequency grid.
    hp_grid : np.ndarray
        Plus polarization grid (n_params x n_freqs), complex.
    param_values : np.ndarray
        Parameter values used.
    """
    hp_list = []
    freqs_common = None

    for val in param_values:
        # Update parameter
        params_dict = fiducial.copy()

        if param_name == "symmetric_mass_ratio":
            # Special handling: vary eta while keeping total mass fixed
            M_total = params_dict["mass_1"] + params_dict["mass_2"]
            m1, m2 = masses_from_total_and_eta(M_total, val)
            params_dict["mass_1"] = m1
            params_dict["mass_2"] = m2
        else:
            params_dict[param_name] = val

        # Generate waveform
        wp = WaveformParameters(
            mass_1=params_dict["mass_1"],
            mass_2=params_dict["mass_2"],
            chi1z=params_dict.get("chi1z", 0.0),
            chi2z=params_dict.get("chi2z", 0.0),
            luminosity_distance=params_dict.get("luminosity_distance", 100.0),
            inclination=params_dict.get("inclination", 0.0),
            phase=params_dict.get("phase", 0.0),
            f_min=f_min,
            f_max=f_max,
            delta_f=delta_f,
            approximant="IMRPhenomXPHM",
        )

        freqs, hp, hc = generate_fd_waveform(wp)

        # Store common frequency grid
        if freqs_common is None:
            freqs_common = freqs

        hp_list.append(hp)

    return freqs_common, np.array(hp_list), param_values


def plot_parameter_sensitivity(
    frequencies: np.ndarray,
    hp_grid: np.ndarray,
    param_values: np.ndarray,
    param_label: str,
    fiducial_name: str,
    output_dir: Path,
    f_plot_min: float = 20.0,
    f_plot_max: float = 300.0,
):
    """
    Create colorbar plots showing how waveform varies with parameter.

    Creates a 2x2 figure with:
    - Log amplitude
    - Unwrapped phase
    - Real part
    - Imaginary part
    """
    # Mask for plotting frequency range
    mask = (frequencies >= f_plot_min) & (frequencies <= f_plot_max)
    freqs_plot = frequencies[mask]

    # Prepare data
    n_params = len(param_values)
    amplitude = np.abs(hp_grid[:, mask])
    phase = np.array([np.unwrap(np.angle(hp[mask])) for hp in hp_grid])
    real_part = hp_grid[:, mask].real
    imag_part = hp_grid[:, mask].imag

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Waveform Sensitivity to {param_label}\n(Fiducial: {fiducial_name})",
        fontsize=14,
    )

    # Color normalization
    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    cmap = cm.viridis

    # Plot each component
    titles = [
        "Log Amplitude",
        "Unwrapped Phase",
        "Real Part",
        "Imaginary Part",
    ]
    data_arrays = [
        np.log10(amplitude + 1e-50),  # Avoid log(0)
        phase,
        real_part,
        imag_part,
    ]
    ylabels = [
        "$\\log_{10}|h_+(f)|$",
        "$\\Phi(f)$ [rad]",
        "$\\mathrm{Re}[h_+(f)]$",
        "$\\mathrm{Im}[h_+(f)]$",
    ]

    for ax, title, data, ylabel in zip(axes.flat, titles, data_arrays, ylabels):
        for i, (row, val) in enumerate(zip(data, param_values)):
            color = cmap(norm(val))
            ax.plot(freqs_plot, row, color=color, alpha=0.7, linewidth=0.8)

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(f_plot_min, f_plot_max)
        ax.grid(True, alpha=0.3)

    # Adjust layout to make room for colorbar on the right
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    # Add colorbar in the reserved space
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(param_label)

    # Save - use full param name for unique filename
    param_safe = param_label.replace("$", "").replace("\\", "").replace(" ", "_")
    param_safe = "".join(c for c in param_safe if c.isalnum() or c == "_")
    filename = f"sensitivity_{fiducial_name}_{param_safe}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"  Saved: {filepath}")


def plot_geometric_frequency_sensitivity(
    frequencies: np.ndarray,
    hp_grid: np.ndarray,
    param_values: np.ndarray,
    param_label: str,
    total_mass: float,
    fiducial_name: str,
    output_dir: Path,
    Mf_min: float = 0.005,
    Mf_max: float = 0.15,
):
    """
    Create colorbar plots in geometric frequency Mf.

    This is the natural coordinate for emulation since waveforms
    become mass-independent in these units.
    """
    # Convert to geometric frequency
    Mf = physical_to_geometric_frequency(frequencies, total_mass)

    # Mask for plotting frequency range
    mask = (Mf >= Mf_min) & (Mf <= Mf_max)
    Mf_plot = Mf[mask]

    # Prepare data
    amplitude = np.abs(hp_grid[:, mask])
    phase = np.array([np.unwrap(np.angle(hp[mask])) for hp in hp_grid])
    real_part = hp_grid[:, mask].real
    imag_part = hp_grid[:, mask].imag

    # Create figure - 2x2 grid like physical frequency plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Waveform in Geometric Frequency: Sensitivity to {param_label}\n"
        f"(Fiducial: {fiducial_name}, M_total = {total_mass:.1f} $M_\\odot$)",
        fontsize=14,
    )

    # Color normalization
    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    cmap = cm.viridis

    # Plot each component
    titles = [
        "Log Amplitude",
        "Unwrapped Phase",
        "Real Part",
        "Imaginary Part",
    ]
    data_arrays = [
        np.log10(amplitude + 1e-50),  # Avoid log(0)
        phase,
        real_part,
        imag_part,
    ]
    ylabels = [
        "$\\log_{10}|h_+(Mf)|$",
        "$\\Phi(Mf)$ [rad]",
        "$\\mathrm{Re}[h_+(Mf)]$",
        "$\\mathrm{Im}[h_+(Mf)]$",
    ]

    for ax, title, data, ylabel in zip(axes.flat, titles, data_arrays, ylabels):
        for i, (row, val) in enumerate(zip(data, param_values)):
            color = cmap(norm(val))
            ax.plot(Mf_plot, row, color=color, alpha=0.7, linewidth=0.8)

        ax.set_xlabel("Geometric Frequency $Mf$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Adjust layout to make room for colorbar on the right
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    # Add colorbar in the reserved space
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(param_label)

    # Save - use full param name for unique filename
    param_safe = param_label.replace("$", "").replace("\\", "").replace(" ", "_")
    param_safe = "".join(c for c in param_safe if c.isalnum() or c == "_")
    filename = f"sensitivity_Mf_{fiducial_name}_{param_safe}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"  Saved: {filepath}")


def plot_time_domain_sensitivity(
    frequencies: np.ndarray,
    hp_grid: np.ndarray,
    param_values: np.ndarray,
    param_label: str,
    total_mass: float,
    fiducial_name: str,
    output_dir: Path,
    t_plot_range: float = 0.1,  # seconds around merger
):
    """
    Create colorbar plots in time domain (both physical and geometric).

    Shows waveform centered on merger.
    """
    import jax.numpy as jnp

    # Get frequency spacing
    delta_f = frequencies[1] - frequencies[0]

    # Color normalization
    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    cmap = cm.viridis

    # Create figure with 2 rows: physical time, geometric time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Time Domain Waveform: Sensitivity to {param_label}\n"
        f"(Fiducial: {fiducial_name}, M_total = {total_mass:.1f} $M_\\odot$)",
        fontsize=14,
    )

    # Process each waveform
    h_td_list = []
    t_list = []
    tM_list = []
    h_tM_list = []

    for hp in hp_grid:
        # Physical time domain
        times, h_td = fd_to_td(jnp.array(hp), delta_f, center=True)
        h_td_list.append(np.array(h_td))
        t_list.append(np.array(times))

        # Geometric time domain - need uniform Mf grid
        Mf = physical_to_geometric_frequency(jnp.array(frequencies), total_mass)
        # Interpolate to uniform Mf grid for FFT
        Mf_uniform, hp_Mf = interpolate_to_uniform_grid(
            jnp.array(hp), Mf, n_points=len(Mf)
        )
        tM, h_tM = geometric_fd_to_td(hp_Mf, Mf_uniform, center=True)
        tM_list.append(np.array(tM))
        h_tM_list.append(np.array(h_tM))

    # Use first waveform's time array as reference
    times = t_list[0]
    tM = tM_list[0]

    # Physical time plots
    t_mask = np.abs(times) < t_plot_range
    ax = axes[0, 0]
    for h_td, val in zip(h_td_list, param_values):
        color = cmap(norm(val))
        ax.plot(times[t_mask] * 1000, h_td[t_mask], color=color, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("$h(t)$")
    ax.set_title("Physical Time Domain")
    ax.grid(True, alpha=0.3)

    # Envelope in physical time
    ax = axes[0, 1]
    for h_td, val in zip(h_td_list, param_values):
        color = cmap(norm(val))
        envelope = np.abs(h_td)  # Hilbert envelope approximation
        ax.semilogy(times[t_mask] * 1000, envelope[t_mask], color=color, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("$|h(t)|$")
    ax.set_title("Envelope (Physical Time)")
    ax.grid(True, alpha=0.3)

    # Geometric time plots - determine appropriate range
    # Convert physical time range to geometric
    MTSUN_SI = 4.925491025543576e-6
    M_seconds = total_mass * MTSUN_SI
    tM_plot_range = t_plot_range / M_seconds  # geometric time range
    tM_mask = np.abs(tM) < min(tM_plot_range, np.max(np.abs(tM)) * 0.8)

    ax = axes[1, 0]
    for h_tM, val in zip(h_tM_list, param_values):
        color = cmap(norm(val))
        ax.plot(tM[tM_mask], h_tM[tM_mask], color=color, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Time $t/M$")
    ax.set_ylabel("$h(t/M)$")
    ax.set_title("Geometric Time Domain (mass-independent)")
    ax.grid(True, alpha=0.3)

    # Envelope in geometric time
    ax = axes[1, 1]
    for h_tM, val in zip(h_tM_list, param_values):
        color = cmap(norm(val))
        envelope = np.abs(h_tM)
        ax.semilogy(tM[tM_mask], envelope[tM_mask], color=color, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Time $t/M$")
    ax.set_ylabel("$|h(t/M)|$")
    ax.set_title("Envelope (Geometric Time)")
    ax.grid(True, alpha=0.3)

    # Adjust layout and add colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(param_label)

    # Save
    param_safe = param_label.replace("$", "").replace("\\", "").replace(" ", "_")
    param_safe = "".join(c for c in param_safe if c.isalnum() or c == "_")
    filename = f"sensitivity_td_{fiducial_name}_{param_safe}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter sensitivity plots for GW waveforms"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/sensitivity",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--fiducial",
        type=str,
        default=None,
        help="Fiducial parameter set to use (default: all)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Parameter Sensitivity Analysis for IMRPhenomXPHM")
    print("=" * 60)

    # Select fiducial sets
    if args.fiducial:
        fiducials = {args.fiducial: FIDUCIAL_PARAMS[args.fiducial]}
    else:
        fiducials = FIDUCIAL_PARAMS

    # Iterate over fiducial parameter sets
    for fid_name, fid_params in fiducials.items():
        print(f"\n--- Fiducial: {fid_name} ---")
        print(f"  m1={fid_params['mass_1']}, m2={fid_params['mass_2']}, "
              f"chi1z={fid_params['chi1z']}, chi2z={fid_params['chi2z']}")

        # Iterate over parameters to vary
        for param_name, (p_min, p_max, n_vals, param_label) in PARAMETER_RANGES.items():
            print(f"\n  Varying: {param_name}")

            # Skip if this parameter matches fiducial (avoid trivial variation)
            if param_name in fid_params:
                # Adjust range to center on fiducial value
                fid_val = fid_params[param_name]
                if param_name.startswith("mass"):
                    # Use wider range for mass
                    param_values = np.linspace(max(5, fid_val * 0.5), fid_val * 1.5, n_vals)
                elif param_name.startswith("chi"):
                    # Center spin range
                    param_values = np.linspace(p_min, p_max, n_vals)
                else:
                    param_values = np.linspace(p_min, p_max, n_vals)
            else:
                param_values = np.linspace(p_min, p_max, n_vals)

            # Generate waveform grid
            try:
                freqs, hp_grid, param_values = generate_waveform_grid(
                    fid_params,
                    param_name,
                    param_values,
                    f_min=10.0,
                    f_max=512.0,
                    delta_f=0.125,
                )

                # Plot in physical frequency
                plot_parameter_sensitivity(
                    freqs, hp_grid, param_values,
                    param_label, fid_name, output_dir,
                    f_plot_min=20.0, f_plot_max=300.0,
                )

                # Plot in geometric frequency
                M_total = fid_params["mass_1"] + fid_params["mass_2"]
                plot_geometric_frequency_sensitivity(
                    freqs, hp_grid, param_values,
                    param_label, M_total, fid_name, output_dir,
                )

                # Plot in time domain (physical and geometric)
                plot_time_domain_sensitivity(
                    freqs, hp_grid, param_values,
                    param_label, M_total, fid_name, output_dir,
                )

            except Exception as e:
                print(f"    Error: {e}")
                continue

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
