# **Advanced Neural Network Architectures for Gravitational Wave Emulation: A Survey of Non-Standard Representation Domains**

## **1\. Introduction**

The dawn of gravitational-wave (GW) astronomy, marked by the historic detection of the binary black hole (BBH) merger GW150914, has fundamentally altered our approach to observing the universe. As the sensitivity of ground-based interferometers—such as Advanced LIGO, Virgo, and KAGRA—continues to improve, the rate of detection is expected to increase by orders of magnitude. This observational success, however, precipitates a significant computational crisis. The extraction of physical parameters from noisy detector data relies on matched filtering and Bayesian inference, techniques that require the generation of millions of distinct waveform templates to map the posterior probability density of the source parameters.

Traditionally, the generation of these waveforms has relied on two primary methods: Numerical Relativity (NR) and semi-analytical approximants. NR simulations, which solve the full non-linear Einstein field equations on adaptive grids, represent the "ground truth" of the field. Yet, they are computationally exorbitant, often requiring thousands of CPU hours for a single merger simulation, rendering them impractical for real-time parameter estimation or large-scale population studies.1 Conversely, semi-analytical approximants (such as the Effective-One-Body (EOB) or Phenom families) offer significantly faster evaluation times but rely on phenomenological tuning and can suffer from accuracy degradation in extreme regions of the parameter space—such as those involving high mass ratios, significant spin-induced precession, or eccentricity.2

In this context, Neural Network (NN) based emulation has emerged as a transformative third pillar. By learning the non-linear mapping between astrophysical parameters and the resulting gravitational strain, neural surrogates can accelerate waveform generation by factors ranging from $10^3$ to $10^5$ while maintaining fidelity comparable to the training data.4 While the initial wave of research focused on simple Multi-Layer Perceptrons (MLPs) mapping parameters directly to 1D time-series vectors or frequency-domain coefficients, the field is currently undergoing a "geometric" revolution. Researchers are increasingly abandoning these standard Euclidean vector spaces in favor of representation domains that better capture the underlying physics and symmetries of the system.

This report provides an exhaustive literature review of these advanced representation domains for NN-based GW emulation. We move beyond standard time and frequency vector spaces to survey:

1. **Time-Frequency Spectrograms and Image Domains:** Leveraging the power of computer vision architectures like U-Nets to handle non-stationary noise and chirp morphologies.7  
2. **Non-Linear Latent Manifolds:** Utilizing autoencoders to discover intrinsic low-dimensional geometric structures (such as spirals) within the waveform parameter space.9  
3. **Physics-Informed Phase Space:** Adopting symplectic and Hamiltonian frameworks to enforce energy conservation and stability in dynamical emulation.11  
4. **Functional Operator Spaces:** employing Fourier Neural Operators (FNOs) to learn resolution-invariant mappings for detector physics and wave propagation.1  
5. **Graph-Theoretic and Equivariant Representations:** Modeling detector networks and simulation meshes as graphs to capture spatial correlations and geometric symmetries.16  
6. **Probabilistic Generative Domains:** Using Diffusion and Score-based models to represent waveforms and noise as trajectories in probability density space.19

The following sections detail the architectures, training strategies, and benchmarking results associated with each of these domains, highlighting how they overcome the limitations of traditional emulation.

## **2\. The Time-Frequency Domain: Spectrograms and Image-Based Architectures**

The transformation of one-dimensional time-series gravitational wave data into two-dimensional time-frequency representations (TFRs) constitutes a significant shift in emulation strategy. This domain transfer is motivated by the distinct morphological characteristics of astrophysical signals versus instrumental noise when viewed in the time-frequency plane. While glitches—transient noise artifacts—often manifest as high-intensity "blobs" or vertical lines spanning wide frequency bands, compact binary coalescences (CBCs) trace characteristic "chirp" patterns with monotonically increasing frequency and amplitude over time.7

### **2.1. U-Net Architectures for Signal Segmentation and Emulation**

The application of Convolutional Neural Networks (CNNs) to TFRs treats GW emulation and detection as an image segmentation or image-to-image translation problem. The U-Net architecture, originally developed for biomedical image segmentation, has become a standard-bearer in this domain due to its unique encoder-decoder structure with skip connections.

Architectural Mechanism:  
The U-Net consists of a contracting path (encoder) that captures context via max-pooling and convolution operations, effectively reducing the spatial dimension while increasing the feature depth. This is symmetric to an expanding path (decoder) that enables precise localization by up-sampling the feature maps. Crucially, skip connections concatenate the high-resolution features from the encoder directly to the decoder. In the context of GW emulation, the input is typically a noisy spectrogram of the detector strain, and the output is a clean probability map or "mask" indicating the pixel-wise presence of a GW signal.8  
Performance and Benchmarking:  
Research utilizing 2D U-Nets has demonstrated robust capability in identifying GW signals from binary black hole (BBH) mergers. For instance, studies simulating BBH mergers with component masses between 7 and 50 $M\_{\\odot}$ report that U-Net algorithms can identify approximately 80% of GW events even in the noisy environment of the third observing run (O3).8  
Unlike traditional matched filtering, which produces a single signal-to-noise ratio (SNR) scalar, the U-Net outputs a time-frequency image of probabilities. This provides a more intuitive, interpretable diagnostic, allowing researchers to visualize exactly which pixels the network attributes to the signal and which to noise.8 This pixel-level granularity is particularly advantageous for identifying "unmodeled" bursts—signals that do not conform perfectly to analytical templates, such as those from supernovae or eccentric binaries—where the network learns generalized chirp morphologies rather than rigid template matching.22  
Training Strategies:  
Training these models typically involves "injection" campaigns. Pure background noise segments are sourced from real detector data (to capture non-Gaussian artifacts), and simulated waveforms (e.g., from IMRPhenom or NR surrogates) are additively injected at varying SNRs. The network minimizes a pixel-wise cross-entropy loss or a Dice coefficient loss, learning to separate the coherent chirp track from the background stochasticity.8

### **2.2. Phase Recovery Challenges and the Griffin-Lim Algorithm**

A fundamental limitation of standard spectrogram-based emulation is the discarding of phase information. A standard spectrogram represents the magnitude of the Short-Time Fourier Transform (STFT), $|STFT(x)|$, essentially ignoring the complex phase angle. However, the phase evolution of a gravitational wave carries critical physical information, including the chirp mass and the precise timing required for coherent combination across a network of detectors.7

The GLA-Grad Architecture:  
To overcome this, recent literature has introduced hybrid architectures that integrate the Griffin-Lim Algorithm (GLA) directly into the deep learning pipeline. The GLA is a classical iterative method that estimates the phase of a signal from its magnitude spectrogram by repeatedly enforcing consistency constraints in the STFT domain.  
In the "GLA-Grad" approach, a deep neural network (often a customized U-Net or autoencoder) is trained to emulate the *magnitude* spectrogram of the GW signal. The output of this network is then passed to a GLA layer—which is often differentiable or executed as a fixed post-processing step—to reconstruct the phase. This "amplitude-first, phase-second" strategy leverages the fact that the amplitude structure of a chirp is smoother and easier for a CNN to learn than the highly oscillatory phase information.7

Benchmarking Results:  
Experimental results on mock injected BBH waveforms (total mass \> 30 $M\_{\\odot}$) indicate that this method yields high consistency in both amplitude and phase alignments. Specifically, the phase recovery is sufficiently accurate to reconstruct the time-domain waveform with high fidelity around the merger stage, where non-linear amplitude dynamics are most pronounced.7 This demonstrates that TFRs, when augmented with phase-recovery algorithms, constitute a complete representation domain for high-fidelity waveform emulation, not just detection.

### **2.3. GWpyxel: Burst Detection in the Pixel Domain**

Complementing the continuous waveform emulation is the "GWpyxel" approach, which focuses on identifying short-duration GW transients (bursts) using mathematical morphology on spectrograms. This pipeline adapts computer vision techniques—specifically Yen’s thresholding method and edge detection—to separate signal clusters from noise. By treating the spectrogram as a pixel map, GWpyxel identifies "connected components" (clusters of pixels) that exhibit the time-frequency connectivity characteristic of astrophysical signals, filtering out disjoint noise pixels. While primarily a detection algorithm, it highlights the utility of the *pixel domain* as a representation space where geometric connectedness encodes physical causality.22

## **3\. Non-Linear Manifold and Latent Space Representations**

Standard reduced-order models (ROMs) in gravitational wave physics typically employ linear dimensionality reduction techniques, such as Singular Value Decomposition (SVD) or Principal Component Analysis (PCA), to construct a basis for the waveform space. While effective, these linear methods often require a relatively large number of basis functions to capture the complex, non-linear dependencies of the waveform on parameters like spin precession and eccentricity. Neural networks offer the capability to discover *non-linear* manifolds—curved hypersurfaces within the high-dimensional data space—that can represent the physics more compactly and efficiently.

### **3.1. Autoencoder-Driven Spiral Representation Learning**

A profound insight into the geometry of the GW parameter space was provided by research using autoencoders to compress the empirical interpolation coefficients of surrogate models. Surrogates typically model the waveform $h(t; \\lambda)$ by interpolating coefficients $c\_i(\\lambda)$ associated with a fixed basis. When researchers trained deep autoencoders to compress these coefficients $c\_i(\\lambda)$ into a 2D latent space, they discovered that the data did not form a disorganized cloud, but rather a coherent **spiral structure**.9

Geometric Interpretation:  
This spiral representation is not an artifact but a reflection of the intrinsic physics of the binary inspiral. The angular position along the spiral was found to be linearly related to the mass ratio ($q$) of the binary system. This suggests that the non-linear variation of the waveform morphology as the mass ratio changes can be topologically mapped to a simple spiral curve in a latent manifold. This is a powerful demonstration of "manifold learning," where the network identifies the single degree of freedom (mass ratio) governing the complexity of the coefficients.10  
Neural Spiral Module:  
Leveraging this discovery, researchers developed a specialized Neural Spiral Module. Instead of using generic fully connected layers, this module explicitly encodes the spiral geometry into the network's architecture. It acts as an inductive bias, forcing the network to learn representations that conform to this discovered topology.

* **Benchmarking:** The integration of the Neural Spiral Module resulted in faster training convergence and improved interpolation accuracy compared to baseline MLPs. The resulting surrogate models achieved state-of-the-art accuracy with evaluation times on the order of milliseconds, far surpassing standard interpolation lookups in speed while maintaining physical faithfulness.9

### **3.2. Conditional Autoencoders (cAEs) for Generative Emulation**

Beyond analyzing coefficients, Conditional Autoencoders (cAEs) and Conditional Variational Autoencoders (cVAEs) utilize the latent domain for the direct generation of time-domain waveforms. In this framework, the encoder compresses a waveform into a latent vector $z$, and the decoder reconstructs it. Crucially, the decoder is conditioned on the physical parameters $\\theta$ (masses, spins), allowing it to generate waveforms from the latent space based on specific physical requests.28

**Performance Metrics:**

* **Overlap Accuracy:** cAE models trained on BBH waveforms have achieved an average overlap (fitting factor) of **\>97%** with matched filtering templates for mass ratios between 1 and 10\. Optimized variants and specific well-trained instances report overlaps as high as **99.5%** for the inspiral-merger phase and **99.74%** for full IMR (Inspiral-Merger-Ringdown) waveforms.28  
* **Generation Speed:** The most significant advantage of this domain is speed. Generating a single waveform via a cAE takes approximately **1 millisecond** (specifically 0.8–1.0 ms in reported benchmarks). This represents a speedup of **10 to 100 times** compared to optimized C-code implementations of analytical models like EOBNRv4 running on the same hardware.6

Training Strategy:  
These models often employ a semi-supervised or self-training strategy. A network is initially trained on a dataset of low-mass-ratio (LMR) waveforms, which are computationally cheaper to simulate. Once trained, the model's generative capability is used to explore the high-mass-ratio (HMR) space, iteratively refining its internal manifold representation. This allows the emulator to generalize effectively to parts of the parameter space (like high mass ratios) that were sparsely represented in the initial ground-truth training set.28

### **3.3. SVD-Based Neural Interpolants**

Another prominent approach in the latent domain combines SVD with deep learning interpolation. Here, the waveform data is first decomposed into SVD spatial modes (basis functions) and time dynamics. Neural networks are then trained to predict the SVD coefficients and the time-series evolution as a function of the astrophysical parameters.29

* **Benchmarking Speed:** When implemented on GPUs, these SVD-NN interpolants can generate batches of $10^4$ waveforms in **$\\lesssim 1$ millisecond**. This extreme throughput is critical for next-generation analyses that may require creating millions of waveform variations to marginalize over uncertainties in real-time.29  
* **Faithfulness:** These models maintain mismatches on the order of $10^{-5}$ for BBH waveforms and $10^{-4}$ for Binary Neutron Star (BNS) waveforms, ensuring they are indistinguishable from the training templates for current detector sensitivities.29

## **4\. Physics-Informed Phase Space and Symplectic Representations**

A recurring challenge in using standard Recurrent Neural Networks (RNNs) or MLPs for dynamic system emulation is their failure to respect conservation laws. A standard NN predicting the orbital trajectory of a binary system will inevitably suffer from numerical dissipation or energy drift over long integration times, eventually violating the physical constraints of the orbit. To address this, researchers have turned to **Phase Space** representations $(q, p)$ governed by **Symplectic** and **Hamiltonian** mechanics.11

### **4.1. Hamiltonian Neural Networks (HNNs)**

Hamiltonian Neural Networks (HNNs) fundamentally alter the learning objective. Instead of learning the vector field of time derivatives $(\\dot{q}, \\dot{p})$ directly, the network learns a scalar function $\\mathcal{H}(q, p; \\theta)$—the Hamiltonian—representing the total energy of the system. The time evolution is then obtained by differentiating this scalar function according to Hamilton's equations:

$$\\frac{dq}{dt} \= \\frac{\\partial \\mathcal{H}}{\\partial p}, \\quad \\frac{dp}{dt} \= \-\\frac{\\partial \\mathcal{H}}{\\partial q}$$

Conservation by Construction:  
By deriving the dynamics from a Hamiltonian, the resulting motion is guaranteed to be symplectic, meaning it preserves the phase space volume (Liouville's theorem) and, for time-independent Hamiltonians, exactly conserves the energy $\\mathcal{H}$. This "conservation by construction" makes HNNs far superior to standard networks for emulating long inspirals where energy conservation is paramount to tracking the orbital phase accurately.12  
**Applications:**

* **Fuzzball Geodesics:** HNNs have been successfully applied to model the motion of massless probes within "fuzzball" geometries (smooth, horizonless microstate geometries in string theory). In these complex, chaotic potentials, HNNs accurately reproduce particle geodesics where traditional integrators might struggle with stability.11  
* **Binary Dynamics:** In BBH systems, HNNs have been used to learn the effective Hamiltonian governing the conservative part of the dynamics. This provides a "shadow Hamiltonian" that allows for stable long-term integration of the orbit.30

### **4.2. Discovering Relativistic Corrections with Universal Differential Equations**

A potent extension of the phase-space approach is the use of Universal Differential Equations (UDEs) to "discover" physics. In this setup, the emulation model is a hybrid: it contains a known physical term (e.g., the Newtonian Hamiltonian) and a neural network term designed to learn the *deviation* from this known physics.31

$$\\dot{x} \= f\_{\\text{Newton}}(x) \+ NN(x)$$

Researchers have demonstrated that when trained on relativistic waveform data, the neural network correctly identifies and learns the Post-Newtonian (PN) relativistic corrections—such as perihelion precession and radiation reaction forces—effectively performing symbolic regression in the phase domain. This allows the emulator to generalize well beyond the training data because it has learned the physical terms (like the 1PN or 2PN potentials) rather than just memorizing trajectory points.31

### **4.3. Symplectic Recurrent Neural Networks (SRNNs)**

While HNNs typically use ODE solvers during inference, Symplectic Recurrent Neural Networks (SRNNs) and their variants (e.g., SympNets) embed symplectic integrators directly into the recurrent layers of a discrete-time network. These networks enforce the symplectic condition on the transition matrix $M$ of the recurrent step: $M^T J M \= J$.  
Benchmarking:  
In tests involving the gravitational N-body problem (which shares dynamical features with binary evolution), SRNNs significantly outperform standard Deep Neural Networks (DNNs). While standard DNNs exhibit rapid energy divergence (errors growing exponentially), symplectic networks maintain bounded energy error over thousands of time steps, making them robust emulators for chaotic or long-duration gravitational dynamics.13

## **5\. Functional Operator Spaces: Learning Physics with Neural Operators**

Standard neural networks map finite-dimensional vectors to finite-dimensional vectors. However, physical fields—such as the gravitational strain $h(t)$ or the spacetime metric $g\_{\\mu\\nu}(x)$—are fundamentally continuous functions. **Neural Operators** are a class of architectures designed to learn mappings between infinite-dimensional function spaces, offering a powerful new domain for GW emulation that is independent of discretization resolution.14

### **5.1. Fourier Neural Operators (FNOs) for Detector Physics**

The Fourier Neural Operator (FNO) has been applied to surrogate the complex physics simulators used for designing GW detectors, such as the software *Finesse*. In this domain, the input is not a waveform parameters vector, but a **function** describing the optical configuration of the interferometer (e.g., spatially varying mirror reflectivities, beam shapes, or heating profiles).1

Architecture and Mechanism:  
The FNO replaces the standard affine transformations of neural networks with a convolution operator defined in the Fourier domain:

$$(K v)(x) \= \\mathcal{F}^{-1} (R \\cdot \\mathcal{F} v)(x)$$

where $\\mathcal{F}$ is the Fourier transform and $R$ is a learnable weight tensor. Because the convolution is performed in frequency space, the operation captures global dependencies (infinite receptive field) and is strictly invariant to the resolution of the spatial grid used to discretize the function.34  
To handle the complex geometry of an interferometer (which is not a simple grid), researchers utilize a Quasi-Universal InterFerOmeter (UIFO) template. The detector design is broken into "patches" (graph nodes representing mirrors/splitters), and the FNO processes these patches using Fourier feature mappings (positional embeddings) to enable the network to learn high-frequency details that standard spectral methods might smooth out.15  
Benchmarking Speed:  
The speedup provided by FNO surrogates in this domain is staggering. Optimization loops for detector design that rely on CPU-based physics simulators can take five days to converge. FNO-based surrogates, leveraging GPU parallelism and auto-differentiation, can explore the same design space and find optimal solutions in hours.35 This capability allows for the "inverse design" of next-generation detectors, where the desired sensitivity curve is input, and the network proposes the optical configuration.

### **5.2. Wave Propagation and Physics-Encoded FNOs**

While primarily demonstrated in seismic and photoacoustic wave equations, FNOs are directly applicable to GW propagation. Research has shown FNOs solving the 2D wave equation orders of magnitude faster than finite-difference time-domain (FDTD) solvers—specifically 26x faster on $64 \\times 64$ grids and scaling better at higher resolutions.14  
Recent innovations include Physics-Encoded FNOs (PeFNOs), which hard-code differential constraints (like the divergence-free condition for certain fields) directly into the operator architecture. Unlike "Physics-Informed" approaches that use loss terms (soft constraints), PeFNOs satisfy the physics exactly by design, which is crucial for emulating the constraints of General Relativity (such as the Bianchi identities) in future full-spacetime emulators.38

## **6\. Graph-Theoretic and Equivariant Representations**

Gravitational wave science is inherently geometric. The detectors form a spatial network on the Earth's surface, and the simulated binaries exist in a 3D space with rotational symmetries. **Graph Neural Networks (GNNs)** and **Equivariant Neural Networks (ENNs)** provide the natural representation domains for these structures.16

### **6.1. Spatiotemporal Graph Neural Networks for Detection**

In the context of multi-detector observation, the data is not just a time series but a signal on a graph. The nodes of the graph represent the observatories (Hanford, Livingston, Virgo), and the edges represent the spatial baselines and light-travel-time correlations.  
Architecture:  
Researchers have developed "Spatiotemporal-Graph AI ensembles" that combine dilated 1D CNNs (to capture temporal history at each node) with GNN layers (to pass messages between nodes). This architecture allows the network to learn "coincidence" naturally—a signal is only valid if consistent messages are passed between detector nodes according to the physical time delays encoded in the edges.16  
Benchmarking:  
This graph-based approach has proven exceptionally scalable. An ensemble of these models running on supercomputers (Polaris/Theta) processed a decade of GW data from a three-detector network in just 3.5 hours, demonstrating the efficiency of representing the detector network as a graph rather than treating streams independently or as a simple stacked vector.16

### **6.2. MeshGraphNets for Simulation**

For emulating the source physics (the binary merger itself), **MeshGraphNets** transform the simulation mesh (finite element or spectral grid) into a graph. Physical variables (fluid density, spacetime curvature) are node attributes, and fluxes are edge messages.

* **Adaptive Resolution:** Unlike CNNs on fixed grids, MeshGraphNets can handle unstructured meshes that adaptively refine resolution near the black holes, where curvature gradients are steepest. This makes them ideal surrogates for the expensive adaptive mesh refinement (AMR) codes used in Numerical Relativity.18

### **6.3. Equivariance and Symmetry-Preserving Networks**

General Relativity is covariant, meaning physical laws are independent of the coordinate system. Standard NNs do not respect this; a rotated binary looks "new" to a standard CNN. **Equivariant Neural Networks** enforce symmetry preservation mathematically.

* **Spherical CNNs:** For analyzing sky localization maps or antenna patterns, Spherical CNNs use filters defined on the sphere ($S^2$) that are equivariant to rotation ($SO(3)$). This ensures that the network's performance is uniform across the sky, avoiding the distortions introduced by projecting the sphere onto 2D planes.39  
* **Transformer-Based Equivariance (Dingo-T1):** The **Dingo-T1** model utilizes a Transformer encoder to process GW data. By treating the data from different detectors as a *set* (or sequence) rather than a fixed vector, and using self-attention, the model becomes permutation-invariant to the ordering of detectors and robust to missing data (e.g., if one detector is offline). This represents a flexible, "set-valued" representation domain that adapts to the heterogeneous configurations of real observing runs.43

## **7\. Generative Probabilistic Domains: Diffusion and Score-Based Models**

The final frontier in representation is the shift from deterministic functions to **probabilistic generative models**. Instead of predicting a single waveform $h(t)$, these models learn the probability density $p(h | \\theta)$ or the noise distribution $p(n)$, typically represented via **Diffusion** processes or **Score-Based** dynamics.20

### **7.1. Score-Based Likelihood Characterization (SLIC)**

The **SLIC** framework addresses a critical limitation in standard GW analysis: the assumption of Gaussian noise. Real detector noise contains non-Gaussian glitches and non-stationary drifts. SLIC employs **Score-Based Generative Models (SGMs)** to learn the gradient of the log-density of the noise, $\\nabla\_n \\log p(n)$, directly from detector data.

* **Mechanism:** The network is trained to denoise data at various noise levels (the diffusion process). This learned "score function" effectively captures the full, non-Gaussian distribution of the noise.  
* **Emulation Capability:** While often discussed in the context of inference, SLIC acts as a high-fidelity *emulator of detector noise*. It can generate noise realizations that are statistically indistinguishable from real data, including complex glitch morphologies. This enables the creation of "mock data" challenges that are far more realistic than those based on Gaussian power spectral densities (PSDs).21

### **7.2. Waveform Synthesis via Wavelet-Diffusion**

Diffusion models are also being applied to generate the waveforms themselves.

* **Wavelet Score-Based Generative Modeling (WSGM):** Generating long time-series data with diffusion models is slow due to the many iterative steps required. WSGM addresses this by operating in the **wavelet domain**. By factorizing the data distribution into wavelet coefficients, the model decouples temporal scales. This allows it to generate high-frequency content (crucial for the merger phase) and low-frequency content (inspiral) efficiently.  
* **Performance:** These models have demonstrated the ability to generate waveforms with frequency content up to 50 Hz (and higher in GW contexts) that capture fine-grained details often smoothed out by VAEs or GANs. The "GLA-Grad" diffusion scheme mentioned in Section 2 is a specific instance of this, combining diffusion with phase recovery to ensure high fidelity.19

## **8\. Comprehensive Benchmarking Analysis**

To summarize the landscape, we compare the discussed architectures across the key metrics of generation speed and physical faithfulness (Table 1).

| Architecture / Domain | Representation Domain | Generation Speed (per waveform/batch) | Faithfulness (Overlap / Mismatch) | Key Advantage |
| :---- | :---- | :---- | :---- | :---- |
| **Traditional EOBNRv4** | Time/Frequency (Analytical) | \~1812 ms (CPU) | Baseline (Reference) | Physical interpretability |
| **Optimized C-Code (EOBNRv4opt)** | Time/Frequency (Optimized) | \~92 ms (CPU) | Baseline | Faster standard CPU code |
| **ANN-Sur / SVD-NN** | Latent SVD / Time | **\~2.7 ms (CPU)**, **\~0.016 ms (GPU batch)** | Mismatch $\\sim 10^{-5}$ (Median) | Extreme throughput for batch analysis |
| **Conditional Autoencoder (cAE)** | Latent Manifold | **\~1 ms (GPU)** | Overlap \> 97%, up to 99.7% | Direct generation from parameters |
| **Neural Spiral Module** | Spiral Latent Manifold | \< 1 ms | High interpolation accuracy | Captures intrinsic mass-ratio geometry |
| **Fourier Neural Operator (FNO)** | Functional / Fourier | Hours vs Days (for optimization) | Resolution Invariant | Accelerates *design* & physics simulation |
| **Hamiltonian NN (HNN)** | Phase Space $(q, p)$ | Slower than cAE (integrator dependent) | **Exact Energy Conservation** | Long-term stability, physical validity |
| **Spatiotemporal GNN** | Graph (Detectors) | 3.5 hrs for 10 years of data | High Coincidence Detection | Scalable to arbitrary detector networks |
| **Score-Based (SLIC)** | Probability Density | Slow (Iterative Sampling) | Captures Non-Gaussian Noise | Realistic noise/glitch emulation |

**Analysis of Results:**

* **Speed:** Latent space models (ANN-Sur, cAE) are the undisputed kings of speed, offering 3-4 orders of magnitude acceleration over standard CPU codes.6 This makes them the primary candidates for real-time alerts and massive population studies.  
* **Faithfulness:** While analytical models are "exact" by definition of the template, Neural Surrogates have reached a level of fidelity (mismatch $10^{-5}$) where they are indistinguishable from the training data for all practical SNR levels in current detectors.6  
* **Stability:** For dynamical simulation (evolving the binary rather than just predicting the strain), Phase Space methods (HNNs/SRNNs) provide the necessary stability that standard RNNs lack, preventing unphysical energy drift.13

## **9\. Conclusion**

The field of gravitational wave emulation has matured rapidly, moving well beyond simple function approximation in the time domain. This review highlights a diversification of **representation domains** driven by the specific physical and computational challenges of GW science.

* **Time-Frequency Spectrograms** have enabled the use of computer vision (U-Nets) to intuitively separate chirps from glitches, providing interpretable visual diagnostics.  
* **Latent Manifolds**—specifically the discovery of "spiral" structures—have proven that the complex phenomenology of binary mergers can be compressed into extremely low-dimensional, geometrically simple spaces, facilitating ultra-fast generation.  
* **Phase Space** representations (HNNs) have solved the problem of physical consistency, allowing neural networks to respect conservation laws and enabling the "discovery" of relativistic corrections.  
* **Operator Learning (FNOs)** and **Graph Representations** have expanded emulation to the functional level (detector design) and the network level (mult-detector coincidence), respectively.

As we look toward the era of 3G detectors (Einstein Telescope, Cosmic Explorer), where overlapping signals and high-SNR requirements will break current analysis pipelines, these advanced representations offer the most promising path forward. By encoding physics into the architecture—whether through symplectic integrators, equivariant layers, or spiral manifolds—these models ensure that the next generation of emulators will be not only fast but rigorously faithful to the laws of General Relativity.

#### **Works cited**

1. Neural surrogates for designing gravitational wave detectors \- arXiv, accessed December 22, 2025, [https://arxiv.org/pdf/2511.19364](https://arxiv.org/pdf/2511.19364)  
2. Fast Waveform Generation for Gravitational Waves using Evolutionary Algorithms \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2404.08576v2](https://arxiv.org/html/2404.08576v2)  
3. Catalog of Simulations of Black Hole Collisions Expands \- Research Impact \- Caltech, accessed December 22, 2025, [https://researchimpact.caltech.edu/research-news/catalog-of-simulations-of-black-hole-collisions-expands](https://researchimpact.caltech.edu/research-news/catalog-of-simulations-of-black-hole-collisions-expands)  
4. Building numerical relativity surrogate models with neural networks: a thesis in Physics. \- UMassD Repository, accessed December 22, 2025, [https://repository.lib.umassd.edu/view/pdfCoverPage?instCode=01MA\_DM\_INST\&filePid=13155671680001301\&download=true](https://repository.lib.umassd.edu/view/pdfCoverPage?instCode=01MA_DM_INST&filePid=13155671680001301&download=true)  
5. Deep learning waveform anomaly detector for numerical relativity catalogs \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2210.07299v2](https://arxiv.org/html/2210.07299v2)  
6. \[2008.12932\] Gravitational-wave surrogate models powered by artificial neural networks: The ANN-Sur for waveform generation \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2008.12932](https://arxiv.org/abs/2008.12932)  
7. Denoising gravitational wave with deep learning in the time-frequency domain \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2511.20731v1](https://arxiv.org/html/2511.20731v1)  
8. Rapid identification of time-frequency domain gravitational wave signals from binary black holes using deep learning, accessed December 22, 2025, [http://202.38.128.72/article/doi/10.1088/1674-1137/ad73ac](http://202.38.128.72/article/doi/10.1088/1674-1137/ad73ac)  
9. Deep Learning Methods for Accelerating Gravitational Wave ..., accessed December 22, 2025, [https://www.researchgate.net/publication/390722318\_Deep\_Learning\_Methods\_for\_Accelerating\_Gravitational\_Wave\_Surrogate\_Modeling](https://www.researchgate.net/publication/390722318_Deep_Learning_Methods_for_Accelerating_Gravitational_Wave_Surrogate_Modeling)  
10. Autoencoder-driven Spiral Representation Learning for Gravitational Wave Surrogate Modelling, accessed December 22, 2025, [https://cidl.csd.auth.gr/resources/journal\_pdfs/Autoencoder-driven%20Spiral%20Representation%20Learning%20for%20Gravitational%20Wave%20Surrogate%20Modelling.pdf](https://cidl.csd.auth.gr/resources/journal_pdfs/Autoencoder-driven%20Spiral%20Representation%20Learning%20for%20Gravitational%20Wave%20Surrogate%20Modelling.pdf)  
11. \[2502.20881\] Hamiltonian Neural Networks approach to fuzzball geodesics \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2502.20881](https://arxiv.org/abs/2502.20881)  
12. Hamiltonian Neural Networks \- NIPS papers, accessed December 22, 2025, [http://papers.neurips.cc/paper/9672-hamiltonian-neural-networks.pdf](http://papers.neurips.cc/paper/9672-hamiltonian-neural-networks.pdf)  
13. \[PDF\] Symplectic Recurrent Neural Networks \- Semantic Scholar, accessed December 22, 2025, [https://www.semanticscholar.org/paper/Symplectic-Recurrent-Neural-Networks-Chen-Zhang/ffb1b305dfd84b81999da356cd8f9790636414df](https://www.semanticscholar.org/paper/Symplectic-Recurrent-Neural-Networks-Chen-Zhang/ffb1b305dfd84b81999da356cd8f9790636414df)  
14. Fourier Neural Operator Network for Fast Photoacoustic Wave Simulations \- MDPI, accessed December 22, 2025, [https://www.mdpi.com/1999-4893/16/2/124](https://www.mdpi.com/1999-4893/16/2/124)  
15. Neural surrogates for designing gravitational wave detectors \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2511.19364](https://arxiv.org/html/2511.19364)  
16. Physics-inspired spatiotemporal-graph AI ensemble for the detection of higher order wave mode signals of spinning binary black hole mergers \- OSTI, accessed December 22, 2025, [https://www.osti.gov/biblio/2369833](https://www.osti.gov/biblio/2369833)  
17. Physics-inspired spatiotemporal-graph AI ensemble for gravitational wave detection, accessed December 22, 2025, [https://www.researchgate.net/publication/371943573\_Physics-inspired\_spatiotemporal-graph\_AI\_ensemble\_for\_gravitational\_wave\_detection](https://www.researchgate.net/publication/371943573_Physics-inspired_spatiotemporal-graph_AI_ensemble_for_gravitational_wave_detection)  
18. Deep Learning Fractal Superconductivity: A Comparative Study of Physics-Informed and Graph Neural Networks Applied to the Fractal TDGL Equation \- MDPI, accessed December 22, 2025, [https://www.mdpi.com/2504-3110/9/12/810](https://www.mdpi.com/2504-3110/9/12/810)  
19. \[2402.15516\] GLA-Grad: A Griffin-Lim Extended Waveform Generation Diffusion Model \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2402.15516](https://arxiv.org/abs/2402.15516)  
20. \[2302.04411\] Geometry of Score Based Generative Models \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2302.04411](https://arxiv.org/abs/2302.04411)  
21. Gravitational-Wave Parameter Estimation in non-Gaussian noise using Score-Based Likelihood Characterization \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/publication/385317849\_Gravitational-Wave\_Parameter\_Estimation\_in\_non-Gaussian\_noise\_using\_Score-Based\_Likelihood\_Characterization](https://www.researchgate.net/publication/385317849_Gravitational-Wave_Parameter_Estimation_in_non-Gaussian_noise_using_Score-Based_Likelihood_Characterization)  
22. Short-Duration Gravitational Wave Burst Detection using Convolutional Neural Network, accessed December 22, 2025, [https://arxiv.org/html/2509.13241v1](https://arxiv.org/html/2509.13241v1)  
23. Denoising gravitational wave with deep learning in the time-frequency domain, accessed December 22, 2025, [https://www.researchgate.net/publication/398025882\_Denoising\_gravitational\_wave\_with\_deep\_learning\_in\_the\_time-frequency\_domain](https://www.researchgate.net/publication/398025882_Denoising_gravitational_wave_with_deep_learning_in_the_time-frequency_domain)  
24. Rapid identification of time-frequency domain gravitational wave signals from binary black holes using deep learning \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2305.19003v2](https://arxiv.org/html/2305.19003v2)  
25. Gravitational Wave-Signal Recognition Model Based on Fourier Transform and Convolutional Neural Network \- PMC \- NIH, accessed December 22, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9536934/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9536934/)  
26. Detection of Gravitational Wave Signals from Precessing Binary Black Hole Systems using Convolutional Neural Network \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2206.12673v2](https://arxiv.org/html/2206.12673v2)  
27. \[2107.04312\] Autoencoder-driven Spiral Representation Learning for Gravitational Wave Surrogate Modelling \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2107.04312](https://arxiv.org/abs/2107.04312)  
28. arXiv:2101.06685v1 \[astro-ph.IM\] 17 Jan 2021, accessed December 22, 2025, [https://arxiv.org/abs/2101.06685](https://arxiv.org/abs/2101.06685)  
29. A neural network-based gravitational wave interpolant with applications to low-latency analyses \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2408.02470v2](https://arxiv.org/html/2408.02470v2)  
30. DeepHMC: a deep-neural-network acclerated Hamiltonian Monte Carlo algorithm for binary neutron star parameter estimation \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2505.02589v1](https://arxiv.org/html/2505.02589v1)  
31. \[2102.12695\] Learning orbital dynamics of binary black hole systems from gravitational wave measurements \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2102.12695](https://arxiv.org/abs/2102.12695)  
32. Discovering the Relativistic Corrections to Binary Black Hole Dynamics \- SciML, accessed December 22, 2025, [https://docs.sciml.ai/Overview/stable/showcase/blackhole/](https://docs.sciml.ai/Overview/stable/showcase/blackhole/)  
33. \[2310.20398\] A hybrid approach for solving the gravitational N-body problem with Artificial Neural Networks \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2310.20398](https://arxiv.org/abs/2310.20398)  
34. Group Equivariant Fourier Neural Operators for Partial Differential Equations \- Proceedings of Machine Learning Research, accessed December 22, 2025, [https://proceedings.mlr.press/v202/helwig23a/helwig23a.pdf](https://proceedings.mlr.press/v202/helwig23a/helwig23a.pdf)  
35. \[2511.19364\] Neural surrogates for designing gravitational wave detectors \- arXiv, accessed December 22, 2025, [https://www.arxiv.org/abs/2511.19364](https://www.arxiv.org/abs/2511.19364)  
36. FOURIER NEURAL OPERATOR SURROGATE MODEL TO PREDICT 3D SEISMIC WAVES PROPAGATION \- Uncecomp 2023, accessed December 22, 2025, [https://2023.uncecomp.org/proceedings/pdf/20362.pdf](https://2023.uncecomp.org/proceedings/pdf/20362.pdf)  
37. Ambient Noise Full Waveform Inversion with Neural Operators \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2503.15013v1](https://arxiv.org/html/2503.15013v1)  
38. A physics-encoded Fourier neural operator approach for surrogate modeling of divergence-free stress fields in solids \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/publication/383494489\_A\_physics-encoded\_Fourier\_neural\_operator\_approach\_for\_surrogate\_modeling\_of\_divergence-free\_stress\_fields\_in\_solids](https://www.researchgate.net/publication/383494489_A_physics-encoded_Fourier_neural_operator_approach_for_surrogate_modeling_of_divergence-free_stress_fields_in_solids)  
39. Geometric deep learning and equivariant neural networks \- research.chalmers.se, accessed December 22, 2025, [https://research.chalmers.se/publication/544317/file/544317\_Fulltext.pdf](https://research.chalmers.se/publication/544317/file/544317_Fulltext.pdf)  
40. SE(3) Equivariant Graph Neural Networks with Complete Local Frames \- Semantic Scholar, accessed December 22, 2025, [https://www.semanticscholar.org/paper/SE(3)-Equivariant-Graph-Neural-Networks-with-Local-Du-Zhang/ae83ca7901aba565604b146911d17ae3ef4d7393](https://www.semanticscholar.org/paper/SE\(3\)-Equivariant-Graph-Neural-Networks-with-Local-Du-Zhang/ae83ca7901aba565604b146911d17ae3ef4d7393)  
41. 433679 PDFs | Review articles in THERMODYNAMICS \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/topic/Thermodynamics/publications/9](https://www.researchgate.net/topic/Thermodynamics/publications/9)  
42. Simulating Complex Particle Dynamics with Graph Neural Networks \- DiVA portal, accessed December 22, 2025, [http://www.diva-portal.org/smash/get/diva2:1984785/FULLTEXT01.pdf](http://www.diva-portal.org/smash/get/diva2:1984785/FULLTEXT01.pdf)  
43. Flexible Gravitational-Wave Parameter Estimation with Transformers \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2512.02968v1](https://arxiv.org/html/2512.02968v1)  
44. \[2402.04384\] Denoising Diffusion Probabilistic Models in Six Simple Steps \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2402.04384](https://arxiv.org/abs/2402.04384)  
45. \[2410.19956\] Gravitational-Wave Parameter Estimation in non-Gaussian noise using Score-Based Likelihood Characterization \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2410.19956](https://arxiv.org/abs/2410.19956)  
46. arXiv:2410.19956v1 \[astro-ph.IM\] 25 Oct 2024, accessed December 22, 2025, [https://arxiv.org/pdf/2410.19956](https://arxiv.org/pdf/2410.19956)  
47. \[2208.05003\] Wavelet Score-Based Generative Modeling \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2208.05003](https://arxiv.org/abs/2208.05003)
