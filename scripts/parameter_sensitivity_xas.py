#!/usr/bin/env python
"""
Parameter sensitivity analysis for IMRPhenomXAS waveforms.

This generates the same plots as parameter_sensitivity.py but using
IMRPhenomXAS (aligned-spin, dominant 22 mode only) for comparison
against IMRPhenomXPHM (higher modes + precession).

Usage:
    python scripts/parameter_sensitivity_xas.py [--output-dir figures/sensitivity_xas]
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
)
from jim_emulators.waveforms.utils import (
    physical_to_geometric_frequency,
    fd_to_td,
    geometric_fd_to_td,
    interpolate_to_uniform_grid,
)


# Fiducial parameter sets - same as XPHM for comparison
# All use aligned spins only (XAS requirement)
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


def format_fiducial_params(fid_params):
    """Format fiducial parameters as a display string."""
    m1 = fid_params["mass_1"]
    m2 = fid_params["mass_2"]
    chi1z = fid_params["chi1z"]
    chi2z = fid_params["chi2z"]
    iota = fid_params["inclination"]

    # Compute derived quantities
    M_total = m1 + m2
    eta = (m1 * m2) / (M_total ** 2)

    return (
        f"$m_1={m1:.1f}M_\\odot$, $m_2={m2:.1f}M_\\odot$, "
        f"$\\chi_{{1z}}={chi1z:.2f}$, $\\chi_{{2z}}={chi2z:.2f}$, "
        f"$\\iota={iota:.2f}$ rad\n"
        f"$M_{{\\rm tot}}={M_total:.1f}M_\\odot$, $\\eta={eta:.3f}$"
    )


def generate_waveform_grid(
    fiducial: dict,
    param_name: str,
    param_values: np.ndarray,
    f_min: float = 10.0,
    f_max: float = 512.0,
    delta_f: float = 0.125,
    approximant: str = "IMRPhenomXAS",
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

        # Generate waveform with XAS
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
            approximant=approximant,
        )

        freqs, hp, hc = generate_fd_waveform(wp)

        # Store common frequency grid
        if freqs_common is None:
            freqs_common = freqs

        hp_list.append(hp)

    return freqs_common, np.array(hp_list), param_values


def plot_unified_sensitivity(
    frequencies: np.ndarray,
    hp_grid: np.ndarray,
    param_values: np.ndarray,
    param_name: str,
    param_label: str,
    fiducial_params: dict,
    fiducial_name: str,
    output_dir: Path,
    f_plot_min: float = 20.0,
    f_plot_max: float = 300.0,
    t_plot_range: float = 0.1,  # seconds around merger
):
    """
    Create a unified 6-panel plot showing all representations.
    """
    import jax.numpy as jnp

    # Frequency masking for plotting
    mask = (frequencies >= f_plot_min) & (frequencies <= f_plot_max)
    freqs_plot = frequencies[mask]
    delta_f = frequencies[1] - frequencies[0]

    # Prepare frequency domain data
    amplitude = np.abs(hp_grid[:, mask])
    phase = np.array([np.unwrap(np.angle(hp[mask])) for hp in hp_grid])
    real_part = hp_grid[:, mask].real
    imag_part = hp_grid[:, mask].imag

    # Prepare time domain data
    h_td_list = []
    times = None
    for hp in hp_grid:
        t, h_td = fd_to_td(jnp.array(hp), delta_f, center=True)
        h_td_list.append(np.array(h_td))
        if times is None:
            times = np.array(t)

    t_mask = np.abs(times) < t_plot_range

    # Create figure - 2 rows x 3 cols
    fig = plt.figure(figsize=(18, 10))

    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                          left=0.05, right=0.88, bottom=0.08, top=0.85,
                          wspace=0.25, hspace=0.3)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    # Title with fiducial parameters - note XAS
    fiducial_str = format_fiducial_params(fiducial_params)
    fig.suptitle(
        f"IMRPhenomXAS: Sensitivity to {param_label}\n"
        f"Fiducial ({fiducial_name}): {fiducial_str}",
        fontsize=12,
        y=0.98,
    )

    # Color normalization
    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    cmap = cm.viridis

    # Panel 1: Log Amplitude
    ax = axes[0]
    for row, val in zip(np.log10(amplitude + 1e-50), param_values):
        ax.plot(freqs_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("$\\log_{10}|h_+(f)|$")
    ax.set_title("Log Amplitude")
    ax.set_xlim(f_plot_min, f_plot_max)
    ax.grid(True, alpha=0.3)

    # Panel 2: Phase
    ax = axes[1]
    for row, val in zip(phase, param_values):
        ax.plot(freqs_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("$\\Phi(f)$ [rad]")
    ax.set_title("Unwrapped Phase")
    ax.set_xlim(f_plot_min, f_plot_max)
    ax.grid(True, alpha=0.3)

    # Panel 3: Time domain waveform
    ax = axes[2]
    for h_td, val in zip(h_td_list, param_values):
        ax.plot(times[t_mask] * 1000, h_td[t_mask], color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("$h(t)$")
    ax.set_title("Time Domain")
    ax.grid(True, alpha=0.3)

    # Panel 4: Real part
    ax = axes[3]
    for row, val in zip(real_part, param_values):
        ax.plot(freqs_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("$\\mathrm{Re}[h_+(f)]$")
    ax.set_title("Real Part")
    ax.set_xlim(f_plot_min, f_plot_max)
    ax.grid(True, alpha=0.3)

    # Panel 5: Imaginary part
    ax = axes[4]
    for row, val in zip(imag_part, param_values):
        ax.plot(freqs_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("$\\mathrm{Im}[h_+(f)]$")
    ax.set_title("Imaginary Part")
    ax.set_xlim(f_plot_min, f_plot_max)
    ax.grid(True, alpha=0.3)

    # Panel 6: Time domain envelope
    ax = axes[5]
    for h_td, val in zip(h_td_list, param_values):
        envelope = np.abs(h_td)
        ax.semilogy(times[t_mask] * 1000, envelope[t_mask], color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("$|h(t)|$")
    ax.set_title("Time Domain Envelope")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.65])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(param_label)

    # Save
    param_safe = param_label.replace("$", "").replace("\\", "").replace(" ", "_")
    param_safe = "".join(c for c in param_safe if c.isalnum() or c == "_")
    filename = f"sensitivity_{fiducial_name}_{param_safe}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"  Saved: {filepath}")


def plot_geometric_unified_sensitivity(
    frequencies: np.ndarray,
    hp_grid: np.ndarray,
    param_values: np.ndarray,
    param_name: str,
    param_label: str,
    fiducial_params: dict,
    fiducial_name: str,
    output_dir: Path,
    Mf_min: float = 0.005,
    Mf_max: float = 0.15,
    t_plot_range: float = 0.1,
):
    """
    Create a unified 6-panel plot in geometric units.
    """
    import jax.numpy as jnp

    M_total = fiducial_params["mass_1"] + fiducial_params["mass_2"]

    # Convert to geometric frequency
    Mf = physical_to_geometric_frequency(frequencies, M_total)

    # Mask for plotting frequency range
    mask = (Mf >= Mf_min) & (Mf <= Mf_max)
    Mf_plot = Mf[mask]

    # Prepare frequency domain data
    amplitude = np.abs(hp_grid[:, mask])
    phase = np.array([np.unwrap(np.angle(hp[mask])) for hp in hp_grid])
    real_part = hp_grid[:, mask].real
    imag_part = hp_grid[:, mask].imag

    # Prepare geometric time domain data
    h_tM_list = []
    tM = None
    for hp in hp_grid:
        Mf_full = physical_to_geometric_frequency(jnp.array(frequencies), M_total)
        Mf_uniform, hp_Mf = interpolate_to_uniform_grid(jnp.array(hp), Mf_full, n_points=len(Mf_full))
        t_M, h_tM = geometric_fd_to_td(hp_Mf, Mf_uniform, center=True)
        h_tM_list.append(np.array(h_tM))
        if tM is None:
            tM = np.array(t_M)

    # Convert physical time range to geometric
    MTSUN_SI = 4.925491025543576e-6
    M_seconds = M_total * MTSUN_SI
    tM_plot_range = t_plot_range / M_seconds
    tM_mask = np.abs(tM) < min(tM_plot_range, np.max(np.abs(tM)) * 0.8)

    # Create figure - 2 rows x 3 cols
    fig = plt.figure(figsize=(18, 10))

    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                          left=0.05, right=0.88, bottom=0.08, top=0.85,
                          wspace=0.25, hspace=0.3)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    # Title with fiducial parameters - note XAS
    fiducial_str = format_fiducial_params(fiducial_params)
    fig.suptitle(
        f"IMRPhenomXAS (Geometric Units): Sensitivity to {param_label}\n"
        f"Fiducial ({fiducial_name}): {fiducial_str}",
        fontsize=12,
        y=0.98,
    )

    # Color normalization
    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    cmap = cm.viridis

    # Panel 1: Log Amplitude
    ax = axes[0]
    for row, val in zip(np.log10(amplitude + 1e-50), param_values):
        ax.plot(Mf_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Frequency $Mf$")
    ax.set_ylabel("$\\log_{10}|h_+(Mf)|$")
    ax.set_title("Log Amplitude")
    ax.grid(True, alpha=0.3)

    # Panel 2: Phase
    ax = axes[1]
    for row, val in zip(phase, param_values):
        ax.plot(Mf_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Frequency $Mf$")
    ax.set_ylabel("$\\Phi(Mf)$ [rad]")
    ax.set_title("Unwrapped Phase")
    ax.grid(True, alpha=0.3)

    # Panel 3: Time domain waveform (geometric)
    ax = axes[2]
    for h_tM, val in zip(h_tM_list, param_values):
        ax.plot(tM[tM_mask], h_tM[tM_mask], color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Time $t/M$")
    ax.set_ylabel("$h(t/M)$")
    ax.set_title("Time Domain (mass-independent)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Real part
    ax = axes[3]
    for row, val in zip(real_part, param_values):
        ax.plot(Mf_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Frequency $Mf$")
    ax.set_ylabel("$\\mathrm{Re}[h_+(Mf)]$")
    ax.set_title("Real Part")
    ax.grid(True, alpha=0.3)

    # Panel 5: Imaginary part
    ax = axes[4]
    for row, val in zip(imag_part, param_values):
        ax.plot(Mf_plot, row, color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Frequency $Mf$")
    ax.set_ylabel("$\\mathrm{Im}[h_+(Mf)]$")
    ax.set_title("Imaginary Part")
    ax.grid(True, alpha=0.3)

    # Panel 6: Time domain envelope (geometric)
    ax = axes[5]
    for h_tM, val in zip(h_tM_list, param_values):
        envelope = np.abs(h_tM)
        ax.semilogy(tM[tM_mask], envelope[tM_mask], color=cmap(norm(val)), alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Geometric Time $t/M$")
    ax.set_ylabel("$|h(t/M)|$")
    ax.set_title("Time Domain Envelope")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.65])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(param_label)

    # Save
    param_safe = param_label.replace("$", "").replace("\\", "").replace(" ", "_")
    param_safe = "".join(c for c in param_safe if c.isalnum() or c == "_")
    filename = f"sensitivity_geometric_{fiducial_name}_{param_safe}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter sensitivity plots for IMRPhenomXAS waveforms"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/sensitivity_xas",
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
    print("Parameter Sensitivity Analysis for IMRPhenomXAS")
    print("(Aligned-spin, dominant 22 mode only)")
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
                    approximant="IMRPhenomXAS",
                )

                # Unified plot in physical units
                plot_unified_sensitivity(
                    freqs, hp_grid, param_values,
                    param_name, param_label,
                    fid_params, fid_name, output_dir,
                    f_plot_min=20.0, f_plot_max=300.0,
                )

                # Unified plot in geometric units
                plot_geometric_unified_sensitivity(
                    freqs, hp_grid, param_values,
                    param_name, param_label,
                    fid_params, fid_name, output_dir,
                )

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
