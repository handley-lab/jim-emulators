# **Accelerating the Gravitational Wave Pipeline: A Comprehensive Review of Neural Network Surrogate Modeling for Precessing Compact Binaries**

## **Abstract**

The detection of gravitational waves (GW) by the Advanced LIGO and Virgo observatories has ushered in a new era of multi-messenger astronomy. However, the scientific potential of these observatories is currently throttled by a severe computational bottleneck: the immense cost of generating accurate theoretical waveforms for complex binary systems, particularly those exhibiting spin-induced orbital precession and higher-order multipole emissions. This report provides an exhaustive analysis of the solution proposed in **"Accelerating multimodal gravitational waveforms from precessing compact binaries with artificial neural networks"** (arXiv:2205.14066) by Thomas, Pratten, and Schmidt. We dissect the proposed technique—hybridizing Reduced Order Modeling (ROM) with Artificial Neural Networks (ANNs)—and evaluate its performance, theoretical underpinnings, and implications. Furthermore, we survey the broader ecosystem of related literature identified in the research query, including applications to eccentric binaries, signal-to-noise ratio (SNR) acceleration, and glitch mitigation, to provide a holistic view of how machine learning is reshaping gravitational wave data analysis.

## ---

**1\. Introduction: The Computational Crisis in Gravitational Wave Astronomy**

### **1.1 The Imperative of Bayesian Inference**

Gravitational wave astronomy is fundamentally an exercise in statistical inference. When a signal is detected by the interferometric network, the immediate scientific priority is to determine the properties of the source system. This process, known as parameter estimation (PE), involves reconstructing the posterior probability density functions (PDFs) for the source parameters—masses, spins, sky location, distance, and orientation—given the noisy data stream.1

The standard methodology for PE relies on stochastic sampling algorithms such as Markov Chain Monte Carlo (MCMC) or Nested Sampling. These algorithms explore the high-dimensional parameter space (typically 15 dimensions for a binary black hole in a quasi-circular orbit) by iteratively comparing theoretical waveform templates against the observed data. To achieve statistically significant posteriors, these samplers require the evaluation of the likelihood function—and thus the generation of a theoretical waveform—on the order of $10^6$ to $10^8$ times per event.1

Consequently, the wall-clock time required for parameter estimation is directly proportional to the computational cost of generating a single waveform. As detector sensitivity improves and the event rate increases, the latency of this analysis becomes a critical limiting factor, potentially delaying electromagnetic follow-up campaigns and hindering population-level studies.

### **1.2 The Complexity of Precession and Higher Modes**

The computational cost of waveform generation is not uniform; it depends heavily on the physical complexity of the model. Early analyses often utilized "aligned-spin" models, where the black hole spins are assumed to be parallel to the orbital angular momentum. This symmetry simplifies the equations of motion, allowing for relatively rapid waveform generation. However, astrophysical reality is rarely so convenient.

General Relativity predicts that generic binary systems will possess misaligned spins, leading to **spin-induced orbital precession**. In such systems, the orbital plane precesses around the total angular momentum vector, causing the gravitational wave signal to exhibit complex amplitude and phase modulations.1 Additionally, while the dominant gravitational radiation is emitted in the quadrupole mode $(\\ell=2, |m|=2)$, asymmetric and precessing systems emit significant energy in **higher-order multipoles** (e.g., $\\ell=3, m=3$ or $\\ell=4, m=4$). Neglecting these effects can lead to substantial systematic biases in the recovered parameters, particularly for unequal-mass binaries or those viewed edge-on.

Modeling these phenomena requires solving the coupled differential equations of the binary's dynamics, a task performed by Effective-One-Body (EOB) codes such as **SEOBNRv4PHM** (Spin-EOB-Numerical Relativity-v4-Precessing-Higher Modes). While SEOBNRv4PHM is highly accurate, incorporating the rigorous physics of precession and higher modes makes it computationally expensive. Generating a single waveform can take seconds.4 When multiplied by the millions of evaluations required for Bayesian inference, the analysis time for a single event can stretch into weeks or months.1

### **1.3 The Rise of Surrogate Modeling**

To address this bottleneck, the community has turned to **surrogate modeling**. A surrogate model is a computationally efficient approximation of a high-fidelity "fiducial" waveform model. Traditional surrogate models, such as Reduced Order Models (ROMs), employ singular value decomposition (SVD) to construct a reduced basis for the waveform space and use standard interpolation techniques (like tensor splines or Gaussian processes) to map physical parameters to basis coefficients.

While effective for lower-dimensional spaces, traditional interpolation methods suffer from the "curse of dimensionality." As the parameter space grows to include precession (adding 3-4 spin dimensions) and eccentricity, the density of training data required for spline interpolation grows exponentially, rendering the model construction intractable.4

The breakthrough presented in arXiv:2205.14066 is the replacement of traditional interpolants with **Artificial Neural Networks (ANNs)**. Neural networks act as universal function approximators capable of handling high-dimensional input spaces without the catastrophic scaling of grid-based interpolation. By combining the rigorous basis projection of ROM with the flexible interpolation of ANNs, Thomas et al. propose a methodology to accelerate the state-of-the-art SEOBNRv4PHM model by orders of magnitude.1

## ---

**2\. Anatomy of the Technique (arXiv:2205.14066)**

The paper "Accelerating multimodal gravitational waveforms from precessing compact binaries with artificial neural networks" details a sophisticated pipeline for creating the **SEOBNN\_v4PHM** surrogate. This section deconstructs the methodology, highlighting the specific architectural choices that enable its performance.

### **2.1 Waveform Decomposition and Coordinate Frames**

A naive application of machine learning might attempt to map binary parameters directly to the time-series strain $h(t)$. However, the oscillatory nature of gravitational waves, combined with their strong dependence on time and phase, makes this approach inefficient. The authors instead employ a physics-informed decomposition strategy that simplifies the learning task.

#### **2.1.1 The Coprecessing Frame**

The central physical insight utilized is the existence of the **coprecessing frame**. In the inertial frame (the detector's frame), a precessing waveform exhibits complex modulations due to the wobbling of the orbital plane. However, if one transforms into a non-inertial frame that tracks the instantaneous orbital plane, the waveform simplifies significantly, resembling that of a non-precessing (aligned-spin) binary.1

The technique decomposes the waveform into:

1. **Coprecessing Modes**: The multipolar waveform components computed in the non-inertial frame. These components vary slowly and smoothly with the physical parameters, making them ideal targets for neural network regression.  
2. **Precession Dynamics**: The time evolution of the Euler angles $\\{\\alpha(t), \\beta(t), \\gamma(t)\\}$ that describe the rotation from the coprecessing frame back to the inertial frame.

By modeling these distinct physical elements separately, the neural networks can learn the smooth behavior of the co-precessing modes without being confused by the rapid modulations of the inertial signal.

### **2.2 Reduced Basis Construction**

Even after simplification, the waveform data consists of thousands of time points. To reduce this dimensionality, the authors employ **Singular Value Decomposition (SVD)**. SVD is a linear algebra technique that identifies an optimal set of orthogonal basis vectors $\\{e\_i(t)\\}$ that span the space of the training waveforms.

For a given waveform mode (e.g., the $(2,2)$ mode), the strain can be approximated as:

$$h\_{\\ell m}(t; \\lambda) \\approx \\sum\_{i=1}^{N} c\_i(\\lambda) e\_i(t)$$

where $\\lambda$ represents the input parameters (masses, spins). The problem of waveform generation is thus reduced to determining the projection coefficients $c\_i(\\lambda)$.1

### **2.3 Neural Network Architecture**

The core innovation of SEOBNN\_v4PHM is the use of Deep Neural Networks to predict these projection coefficients.

#### **2.3.1 Network Topology**

The authors implement separate neural networks for the amplitude and phase of each coprecessing mode, as well as for the Euler angles governing the precession. The networks are typically **Multi-Layer Perceptrons (MLPs)**, consisting of:

* **Input Layer**: Accepts the 4-dimensional parameter vector $\\{q, \\chi\_1, \\chi\_2, \\chi\_{eff}\\}$, where $q$ is the mass ratio and $\\chi$ are the spin components.  
* **Hidden Layers**: Several fully connected layers with non-linear activation functions (likely ReLU, Tanh, or Swish). The depth and width of these layers are determined through hyperparameter optimization to balance expressivity with evaluation speed.  
* **Output Layer**: Outputs the coefficients $c\_i$ for the reduced basis or the values at empirical interpolation nodes.

#### **2.3.2 Empirical Interpolation vs. Direct Coefficient Prediction**

A critical methodological detail is the use of **Empirical Interpolation (EI)**. Rather than predicting the abstract basis coefficients $c\_i$ directly, the network is often trained to predict the waveform's value at specific "time nodes" $T\_j$. These physical values are then projected onto the global basis. This approach is generally more robust, as the network learns to predict physical quantities (amplitude/phase at a specific time) rather than mathematical abstractions, leading to better generalization across the parameter space.1

### **2.4 Training and Validation**

The surrogate is trained on a dataset generated by the fiducial SEOBNRv4PHM code. The training process involves minimizing a loss function, typically the Mean Squared Error (MSE) between the predicted and true waveform values.

Validation is performed by calculating the mismatch (or infidelity) $\\mathcal{M}$ against a hold-out test set. The mismatch is defined as:

$$\\mathcal{M} \= 1 \- \\max\_{t\_c, \\phi\_c} \\frac{\\langle h\_{surr} | h\_{fid} \\rangle}{\\sqrt{\\langle h\_{surr} | h\_{surr} \\rangle \\langle h\_{fid} | h\_{fid} \\rangle}}$$

where $\\langle \\cdot | \\cdot \\rangle$ denotes the inner product weighted by the detector's noise power spectral density (PSD). A mismatch of $\\mathcal{M} \< 10^{-3}$ is generally required for the model to be indistinguishable from the truth at current detector sensitivities. The authors report median mismatches well below this threshold, confirming the surrogate's fidelity.4

## ---

**3\. Performance Analysis: Benchmarking Speed and Accuracy**

The primary motivation for developing SEOBNN\_v4PHM is computational acceleration. The results presented in the paper and subsequent citations demonstrate that the technique achieves this goal with remarkable success.

### **3.1 Computational Speed-up**

The performance gains are analyzed across different hardware architectures, specifically Central Processing Units (CPUs) and Graphics Processing Units (GPUs).

#### **3.1.1 CPU Performance**

On a standard CPU, the generation of a single waveform using the fiducial SEOBNRv4PHM code takes approximately **2 to 5 seconds** (2000-5000 ms), depending on the total mass and starting frequency. In contrast, the neural surrogate SEOBNN\_v4PHM generates a waveform in approximately **18 milliseconds**. This represents a speed-up factor of roughly **100x to 200x**.1

This acceleration alone is sufficient to reduce the runtime of a standard Bayesian inference job from weeks to hours, making it feasible to run parameter estimation on laptop-class hardware.

#### **3.1.2 GPU Performance and Batching**

The architecture of neural networks is inherently parallel, consisting primarily of matrix-vector multiplications. This makes them ideally suited for GPU acceleration. When running on a GPU, the surrogate can generate a single waveform in roughly **0.5 ms**.

However, the true power of the GPU implementation lies in **batch generation**. By vectorizing the input, the surrogate can generate thousands of waveforms simultaneously. The authors report that for batch sizes of $10^4$, the amortized time per waveform drops to **less than 0.01 ms** (or 10 microseconds). This corresponds to a speed-up factor exceeding **100,000x** relative to the serial execution of the fiducial code.1

| Hardware | Model | Execution Mode | Time per Waveform | Speed-up Factor |
| :---- | :---- | :---- | :---- | :---- |
| **CPU** | SEOBNRv4PHM (Fiducial) | Serial | \~2000 \- 5000 ms | 1x (Baseline) |
| **CPU** | **SEOBNN\_v4PHM** | Serial | **\~18 ms** | **\~100x \- 200x** |
| **GPU** | **SEOBNN\_v4PHM** | Single | **\~0.5 ms** | **\~4000x** |
| **GPU** | **SEOBNN\_v4PHM** | Batch ($10^4$) | **\< 0.01 ms** | **\> 100,000x** |

Table 1: Comparative computational performance metrics for waveform generation.1

### **3.2 Accuracy and Robustness**

Speed is of little value if the waveforms are inaccurate. The paper validates the surrogate extensively across the parameter space, including regions with high mass ratios and high spin magnitudes where precession effects are most pronounced.

The reported mismatches are consistently low, with median values around $10^{-4}$ to $10^{-5}$. This level of accuracy ensures that the systematic errors introduced by the neural network approximation are significantly smaller than the statistical uncertainties inherent in the data (due to detector noise). Consequently, parameter estimation results obtained with the surrogate are statistically identical to those obtained with the slow fiducial model.2

## ---

**4\. The Ecosystem of Neural Acceleration: Related Papers and Applications**

The user query requested a search for "papers relating to this technique." The research snippets reveal a rich ecosystem of work that either extends the methodology of 2205.14066 or applies similar neural acceleration techniques to adjacent problems in gravitational wave physics. This section synthesizes these related works, categorizing them by their specific application.

### **4.1 Extending to Eccentricity: SEOBNRE\_AIq5e2**

While 2205.14066 focused on quasi-circular binaries, many astrophysical formation channels (such as dynamical capture in dense stellar clusters) predict the existence of binaries with non-negligible orbital **eccentricity**. Eccentricity introduces additional timescales and modulations to the waveform, making it even more challenging to model than precession.

The papers

4

and

6

describe **SEOBNRE\_AIq5e2**, a neural surrogate designed specifically for eccentric, spin-aligned binary black holes.

* **Technique**: Similar to the Thomas et al. paper, this model uses a combination of data resampling and neural networks. It incorporates an "adaptive resampling" technique during training to handle the complex morphology of eccentric waveforms.  
* **Performance**: The model achieves a generation speed of **4.3 ms per waveform** on a CPU, with a mean mismatch of $1.02 \\times 10^{-3}$.  
* **Relevance**: This work demonstrates the scalability of the "Neural Surrogate" paradigm. If ANNs can handle the chaotic dynamics of eccentricity, they are likely robust enough for almost any signal morphology predicted by General Relativity.6

### **4.2 Emulating Numerical Relativity: The "Deep Learning Powered" Surrogate**

Snippet

7

references a paper titled **"Deep learning powered numerical relativity surrogate for binary black hole waveforms"** (Freitas et al., 2025). This represents the logical endpoint of the surrogate modeling trajectory.

* **Context**: Numerical Relativity (NR) simulations solve the full Einstein field equations and are the most accurate models available. However, a single simulation can take months on a supercomputer.  
* **Technique**: This paper applies the SVD \+ ANN architecture directly to NR data. It describes a "2-stage training approach" to ensure stability and accuracy.  
* **Implication**: By emulating NR directly, rather than an EOB approximation, this approach bypasses the systematic errors of the EOB formalism entirely. It offers the holy grail of waveform modeling: NR accuracy at millisecond speeds.7

### **4.3 Accelerating SNR Computation and Search Pipelines**

The acceleration provided by neural networks is not limited to parameter estimation; it is also revolutionizing the **search** phase (detection).

* Snippet 9 (arXiv:2408.02470): This paper, "Accelerating and signal-to-noise ratio (SNR) computation...", applies ANNs to the calculation of the SNR time series itself.  
* **Mechanism**: The matched-filter search involves correlating the data with a template bank. This operation is essentially an inner product. By using neural networks to interpolate the SNR manifold directly, or to generate the waveform components for the filter, this work achieves computation times of **6 ms on CPU** and **0.4 ms on GPU**.  
* **Application**: This speed allows for denser template banks, improving the "match" with potential signals and thus increasing the detection volume (and event rate) of the observatory.9

### **4.4 Glitch Mitigation with Convolutional Neural Networks**

Data quality is as important as waveform accuracy. Gravitational wave detectors are plagued by non-Gaussian noise transients known as "glitches."

* Snippet 10 (arXiv:2410.15513): "Mitigating the impact of noise transients... using reduced basis timeseries and convolutional neural networks."  
* **Connection**: While 2205.14066 uses MLPs for waveform generation, this paper uses **Convolutional Neural Networks (CNNs)** for signal classification. It projects the data onto a reduced basis (similar to the SVD step in surrogates) and treats the coefficients as an "image" for the CNN.  
* **Result**: The integration of this technique into a search pipeline increased the true positive rate by 4-7% while suppressing false positives. This highlights how the dimensionality reduction techniques central to surrogate modeling (SVD) are also enabling better noise rejection.10

### **4.5 Frequency-Domain Approximants: IMRPhenomXODE**

Snippet

7

mentions **IMRPhenomXODE**, a phenomenological frequency-domain model.

* **Comparison**: Unlike the time-domain EOB models (which solve differential equations), phenomenological models are analytic fits to hybrid data. They are naturally fast but often lack the full physical fidelity of the EOB dynamics.  
* **Relevance**: The neural surrogate (SEOBNN) bridges the gap between these two families. It offers the physics of the EOB model (which IMRPhenom attempts to approximate) but with the evaluation speed of the phenomenological model. The existence of IMRPhenomXODE highlights the competitive landscape of waveform modeling, where multiple approaches strive to balance speed and accuracy.7

## ---

**5\. Technical Synthesis: Methodological Insights**

Analyzing the collective findings of these papers reveals several key technical insights regarding the application of machine learning to physics.

### **5.1 The Shift to "Data-Driven" Physics**

The transition from SEOBNRv4PHM (Code) to SEOBNN\_v4PHM (Network) represents a fundamental shift in scientific computing. Traditionally, acceleration was achieved by optimizing the solver (e.g., better integrators, C++ vectorization). The neural surrogate approach abandons the solver entirely at runtime, relying instead on emulating the solution.  
This implies that the limit on waveform generation speed is no longer dictated by the complexity of the differential equations (General Relativity), but by the inference speed of the neural network architecture (Matrix Multiplication). Since matrix multiplication is the primary workload of modern AI hardware, gravitational wave physics is effectively "drafting" behind the massive industrial optimization of AI accelerators.1

### **5.2 The Importance of Physics-Informed Inductive Bias**

A recurring theme across all successfully applied papers 6 is the refusal to treat the problem as a generic "black box" regression.

* **Decomposition**: The use of the coprecessing frame and Euler angles in 2205.14066 is a physics-informed choice that simplifies the learning landscape.  
* **SVD Basis**: Using SVD rather than allowing the network to output raw time series enforces a linear constraint that guarantees the output looks like a gravitational wave.  
* **Implication**: The most successful "AI for Science" models are those that bake domain knowledge into the architecture or data preprocessing, rather than relying on the network to learn the laws of physics from scratch.1

### **5.3 Hardware-Software Co-design**

The explicit optimization for GPUs mentioned in 1 and 4 highlights the necessity of hardware-aware programming. The massive speed-ups (\>100,000x) are only possible because the algorithm (neural network inference) aligns perfectly with the hardware architecture (Tensor Cores). This suggests that future waveform models will likely be designed primarily as tensor operations to exploit next-generation AI accelerators.

## ---

**6\. Broader Implications for the Future of the Field**

The technique pioneered in 2205.14066 is not merely a tool for current analyses; it is a critical enabler for the future of the field.

### **6.1 Enabling 3rd Generation Detectors**

The next generation of GW observatories, such as the **Einstein Telescope (ET)** and **Cosmic Explorer (CE)**, will have sensitivities extending down to lower frequencies (e.g., 5 Hz). A signal starting at 5 Hz stays in the detector band for minutes or hours, compared to seconds for current detectors.

* **The Scaling Problem**: Generating a waveform that is hours long using an ODE integrator is prohibitively expensive and prone to accumulated phase errors.  
* **The Surrogate Solution**: Neural surrogates, which map parameters directly to basis coefficients, can generate these long waveforms nearly as instantly as short ones, provided the basis is robust. The batching capability on GPUs will be essential to handle the high event rates (potentially hundreds per day) expected in the 3G era.2

### **6.2 Global Parameter Estimation and Population Studies**

Currently, parameter estimation is performed on a per-event basis. However, questions about the formation channels of black holes require analyzing the population as a whole.  
The extreme speed of the neural surrogate allows for hierarchical Bayesian inference where the population hyperparameters and individual event parameters are sampled simultaneously. It also enables large-scale tests of General Relativity, where thousands of alternative theories can be checked against the data in a reasonable timeframe.11

### **6.3 Low-Latency Multi-Messenger Astronomy**

When a binary neutron star merger is detected, every second counts. Astronomers need to know the sky location and distance immediately to point their telescopes. The CPU speed-up provided by SEOBNN\_v4PHM means that full Bayesian parameter estimation (including precession) can potentially be run in near-real-time, providing high-quality alerts to the astronomical community within minutes of the merger.1

## ---

**7\. Conclusion**

The research presented in arXiv:2205.14066, **"Accelerating multimodal gravitational waveforms from precessing compact binaries with artificial neural networks,"** constitutes a milestone in gravitational wave data analysis. By rigorously combining reduced order modeling with deep learning, the authors have successfully decoupled the computational cost of waveform generation from the physical complexity of the model.

**Summary of Key Findings:**

1. **Technique**: A hybrid architecture using SVD for dimensionality reduction and Deep Neural Networks for coefficient interpolation, underpinned by a physics-informed decomposition into the coprecessing frame.  
2. **Performance**: The method delivers a **\~200x speed-up on CPUs** and a staggering **\~100,000x speed-up on GPUs** (amortized), reducing analysis times from weeks to hours.1  
3. **Fidelity**: The surrogate maintains a mismatch of $\\mathcal{M} \\sim 10^{-4}$, ensuring indistinguishability from the trusted fiducial model.2  
4. **Ecosystem**: This work is part of a rapidly expanding family of neural acceleration techniques, now covering eccentric binaries 6, numerical relativity emulation 7, and SNR computation.9

As gravitational wave astronomy transitions from a regime of discovery to one of precision measurement and population statistics, the computational efficiency provided by techniques like SEOBNN\_v4PHM will be as vital as the sensitivity of the detectors themselves. This work firmly establishes Machine Learning not just as an auxiliary tool, but as a central pillar of the modern physics workflow.

---

**End of Report**

#### **Works cited**

1. arXiv:2205.14066v3 \[gr-qc\] 24 Nov 2022, accessed December 22, 2025, [https://arxiv.org/pdf/2205.14066](https://arxiv.org/pdf/2205.14066)  
2. Accelerating multimodal gravitational waveforms from precessing compact binaries with artificial neural networks \- University of Birmingham's Research Portal, accessed December 22, 2025, [https://research.birmingham.ac.uk/files/189047420/ThomasL2022Accelerating.pdf](https://research.birmingham.ac.uk/files/189047420/ThomasL2022Accelerating.pdf)  
3. Lucy M. Thomas's scientific contributions \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/scientific-contributions/Lucy-M-Thomas-2177054908](https://www.researchgate.net/scientific-contributions/Lucy-M-Thomas-2177054908)  
4. Gravitational-wave surrogate models powered by artificial neural networks \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/publication/349983994\_Gravitational-wave\_surrogate\_models\_powered\_by\_artificial\_neural\_networks](https://www.researchgate.net/publication/349983994_Gravitational-wave_surrogate_models_powered_by_artificial_neural_networks)  
5. \[2205.14066\] Accelerating multimodal gravitational waveforms from precessing compact binaries with artificial neural networks \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2205.14066](https://arxiv.org/abs/2205.14066)  
6. Rapid Eccentric Spin-Aligned Binary Black Hole Waveform Generation Based on Deep Learning \- Qeios, accessed December 22, 2025, [https://www.qeios.com/read/MFL8ZK](https://www.qeios.com/read/MFL8ZK)  
7. \[PDF\] Accelerating multimodal gravitational waveforms from ..., accessed December 22, 2025, [https://www.semanticscholar.org/paper/ef4f3ef8b16300a02aef6189756f22f1abf1636c](https://www.semanticscholar.org/paper/ef4f3ef8b16300a02aef6189756f22f1abf1636c)  
8. Numerical relativity surrogate model with memory effects and post-Newtonian hybridization, accessed December 22, 2025, [https://par.nsf.gov/servlets/purl/10475413](https://par.nsf.gov/servlets/purl/10475413)  
9. Accelerated signal-to-noise ratio interpolation across the gravitational-wave signal manifold powered by artificial neural networks \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2408.02470v1](https://arxiv.org/html/2408.02470v1)  
10. Mitigating the impact of noise transients in gravitational-wave searches using reduced basis timeseries and convolutional neural networks \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2410.15513v1](https://arxiv.org/html/2410.15513v1)  
11. \[PDF\] New effective precession spin for modeling multimodal gravitational waveforms in the strong-field regime | Semantic Scholar, accessed December 22, 2025, [https://www.semanticscholar.org/paper/324e4a5cbdd42e10b29b204cde63133cbbe05aff](https://www.semanticscholar.org/paper/324e4a5cbdd42e10b29b204cde63133cbbe05aff)
