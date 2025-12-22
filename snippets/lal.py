import lal
from lal import MSUN_SI, MTSUN_SI, PC_SI
import lalsimulation as lalsim
from lalsimulation import IMRPhenomXPHM
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from jax.scipy.integrate import trapezoid

MPC_SI = PC_SI * 1e6
print(lalsim.__version__)

psd_freqs, *psd_arrs = np.loadtxt("psds/ET-D-psd.txt", unpack=True)
psd_arr = psd_arrs[0]


def get_freq_array(sampling_freqs, duration):
    # Build the frequency grid
    delta_t = 1 / sampling_freqs
    tlen = int(round(duration * sampling_freqs))
    freqs = jnp.fft.rfftfreq(tlen, delta_t)
    return freqs


def get_nyquist_mask(frequencies):
    """Create a mask that zeros the last NYQUIST_BINS_TO_ZERO frequency bins."""
    n_freqs = len(frequencies)
    return jnp.where(jnp.arange(n_freqs) < n_freqs - NYQUIST_BINS_TO_ZERO, 1.0, 0.0)


def noise_weighted_inner_product(h1, h2, psd, frequencies):
    integrand = jnp.conj(h1) * h2 / psd
    return 4 * trapezoid(integrand, x=frequencies, axis=-1).real


def compute_match(h1, h2, psd, frequencies):
    h1_sq = noise_weighted_inner_product(h1, h1, psd, frequencies)
    h2_sq = noise_weighted_inner_product(h2, h2, psd, frequencies)
    h1_h2 = noise_weighted_inner_product(h1, h2, psd, frequencies)
    match = h1_h2 / jnp.sqrt(h1_sq * h2_sq)
    return match.real


s = 4096

# Yes, I should have combined the XAS and tidal functions, I am lazy.


def lal_XPHM_waveform(
    # not implemented args, just to show how it is called
):

    lal_hpc = lalsim.SimInspiralChooseFDWaveform(
        mass_1 * MSUN_SI,
        mass_2 * MSUN_SI,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        luminosity_distance,  # distance (can set to 1 Mpc for normalisation
        inclination,
        phase,
        0.0,  # longitude ascending nodes
        0.0,  # eccentricity
        0.0,  # mean_per_ano
        1.0 / duration,  # delta_frequency
        f_min,  # min frequency
        f_max,  # max frequency
        reference_frequency,  # reference frequency
        None,  # waveform dictionary (lal.Dict)
        IMRPhenomXPHM,  # approximant (lalsim approximant)
    )
    hpc_len = len(lal_hpc[0].data.data)

    frequencies = get_freq_array(sampling_freqs=fs, duration=duration)
    zeros = jnp.zeros_like(frequencies, dtype=jnp.complex128)
    hp = zeros.at[:hpc_len].set(lal_hpc[0].data.data)
    hc = zeros.at[:hpc_len].set(lal_hpc[1].data.data)

    return frequencies, hp, hc


duration = 16.0
f_min = 5.0
f_max = fs / 2

rand_arr = jax.random.uniform(
    jax.random.PRNGKey(seed), (8, Nsamples), dtype=jnp.float64
)
total_mass = rand_arr[0] * 1.0 + 0.5
mass_ratio = rand_arr[1] * 0.95 + 0.05
mass_1 = total_mass / (1 + mass_ratio)
mass_2 = total_mass - mass_1
spin_1z, spin_2z = rand_arr[2:4] * 2 - 1
inclination = rand_arr[4] * jnp.pi
phase = rand_arr[5] * 2 * jnp.pi

frequencies = get_freq_array(sampling_freqs=fs, duration=duration)
interp_psd = jnp.interp(frequencies, psd_freqs, psd_arr)
