# **Advanced Machine Learning Surrogate Architectures for Gravitational Waveform Generation: A Comprehensive Analysis of PCA-Based and Neural Network-Enhanced Methodologies**

## **1\. Executive Summary**

The detection of gravitational waves (GW) by the LIGO-Virgo-KAGRA (LVK) collaboration has ushered in a new era of multi-messenger astronomy. However, the scientific maximization of these detections is currently bottlenecked by the computational cost of Bayesian parameter estimation (PE). Traditional methods require the generation of millions of waveform templates, a task that is computationally prohibitive when using high-fidelity Numerical Relativity (NR) simulations or complex Effective-One-Body (EOB) approximants, particularly when including higher-order modes (HOMs) and precession.

This report provides an exhaustive analysis of a specific class of solution: **Time-Domain, PCA-Based Neural Network Surrogates**. This methodology, exemplified by the reference work *Generating Higher Order Modes from Binary Black Hole mergers with Machine Learning* (Grimbergen et al., 2024), represents a paradigm shift from analytical approximation to data-driven emulation.

The core technique identified involves a tripartite architecture:

1. **Decomposition:** Separating the complex waveform strain into amplitude and phase components for each spherical harmonic mode.  
2. **Dimensionality Reduction:** Utilizing Principal Component Analysis (PCA) or Singular Value Decomposition (SVD) to project high-dimensional time-series data onto a compact linear basis.  
3. **Regression:** Employing Artificial Neural Networks (ANNs)—typically Deep Multi-Layer Perceptrons (MLPs)—to map the intrinsic physical parameters of the binary (mass ratio, spins) to the coefficients of the reduced basis.

This report surveys the research landscape for papers utilizing this specific "Grimbergen Technique" and its derivatives. We identify and dissect key parallel efforts, including the **mlgw framework** (Schmidt et al.), which established the foundational pipeline; **NRSurNN** (Freitas et al.), which introduces transfer learning to train directly on sparse NR data; and **SEOBNRE\_AI**, which adapts the architecture for eccentric binaries via adaptive resampling.

Our analysis reveals that this architecture has become a dominant standard for next-generation waveform modeling, offering speedups of $\\mathcal{O}(10^2)$ to $\\mathcal{O}(10^4)$ while maintaining mismatches as low as $10^{-4}$ against fiducial models. We further explore emerging variations utilizing Transformer architectures and Fourier Analysis Networks (FANs), which challenge the PCA-based status quo. The integration of these models into inference pipelines like bilby suggests a future where real-time, high-fidelity parameter estimation is driven entirely by neural surrogates.

## **2\. The Reference Framework: Deconstructing Grimbergen et al. (2402.06587)**

To accurately identify and compare related works, we must first rigorously define the "technique" presented in the reference paper *Generating Higher Order Modes from Binary Black Hole mergers with Machine Learning*.1 The significance of this work lies not just in its use of machine learning, but in the specific structural choices made to handle the complexity of General Relativity solutions in the time domain.

### **2.1 The Computational Problem: Higher Order Modes (HOMs)**

Standard gravitational wave analysis often relies on the dominant quadrupolar mode $(\\ell, m) \= (2, \\pm 2)$. However, for systems with unequal masses or high inclinations, Higher Order Modes (e.g., $(3,3), (4,4)$) become crucial for breaking degeneracies between distance and inclination and for testing General Relativity.3

Calculating these modes using the state-of-the-art EOB approximant **SEOBNRv4HM** is computationally expensive. The authors of 2402.06587 aim to create a surrogate that mimics SEOBNRv4HM but operates orders of magnitude faster.

### **2.2 The Technical Pipeline**

The "Grimbergen Technique" can be formalized as a specific sequence of operations:

1. Spherical Harmonic Decomposition:  
   The polarizations $h\_+(t)$ and $h\_\\times(t)$ are expanded:

   $$h(t, \\iota, \\varphi\_0) \= \\sum\_{\\ell, m} {}^{-2}Y\_{\\ell m}(\\iota, \\varphi\_0) h\_{\\ell m}(t)$$

   This isolates the angular dependence, leaving the machine learning model to learn only the time-dependent coefficients $h\_{\\ell m}(t)$.2  
2. Amplitude-Phase Decomposition (The Ansatz):  
   Directly learning the oscillatory function $h\_{\\ell m}(t)$ is difficult for neural networks due to the high-frequency features. The authors decompose each mode into amplitude $A\_{\\ell m}$ and phase $\\phi\_{\\ell m}$:

   $$h\_{\\ell m}(t) \= A\_{\\ell m}(t) e^{-i\\phi\_{\\ell m}(t)}$$

   Crucially, these functions are smooth and slowly varying across the parameter space, making them ideal targets for regression.2  
3. Linear Dimensionality Reduction (PCA):  
   The time-series data for $A$ and $\\phi$ (discretized on a grid) lives in a high-dimensional vector space $\\mathbb{R}^N$. The authors apply Principal Component Analysis (PCA) to find an orthonormal basis $\\{V\_k(t)\\}$ that captures the variance of the dataset.

   $$X(t; \\theta) \\approx \\sum\_{k=1}^{K} c\_k(\\theta) V\_k(t)$$

   This reduces the problem from predicting $N$ time points to predicting $K$ coefficients (where $K \\ll N$). The reference paper notes using PCA to reduce dimensionality significantly while retaining high fidelity.2  
4. Neural Network Regression:  
   An ensemble of Artificial Neural Networks (ANNs) is trained to approximate the map $f: \\theta \\rightarrow \\{c\_k\\}$. The inputs $\\theta$ are the mass ratio $q$ and dimensionless spins $\\chi\_{1z}, \\chi\_{2z}$.

### **2.3 Performance Benchmarks**

The reference paper establishes a benchmark for this technique:

* **Faithfulness:** Median mismatch of $10^{-4}$ against the training approximant.1  
* **Speed:** Generation of a single waveform is two orders of magnitude faster than the SEOBNRv4HM code.  
* **Scalability:** Batch generation on GPUs increases this speedup factor significantly.1

This defines the search criteria for our report: We are looking for papers that use **PCA/SVD decomposition** of **time-domain waveforms** coupled with **Neural Network regression** to accelerate **waveform generation**.

## **3\. The mlgw Lineage: The Foundational Precursor**

The most direct relative to the work of Grimbergen et al. is the mlgw (Machine Learning Gravitational Waves) project, led by Stefano Schmidt (a co-author on the reference paper). The mlgw framework appears in multiple snippets 5 and represents the genesis of this specific architectural approach.

### **3.1 mlgw: Machine Learning Gravitational Waves from Binary Black Hole Mergers**

The paper *Machine Learning Gravitational Waves from Binary Black Hole Mergers* (Schmidt et al., Phys. Rev. D 103, 043020, 2021\) 5 is the seminal work in this specific lineage.

#### **Technical Alignment**

The methodology in mlgw is nearly identical to 2402.06587, confirming it as the primary precursor:

* **Decomposition:** It utilizes the same amplitude/phase splitting.  
* **Dimensionality Reduction:** It employs PCA to handle the dimensionality of the time-domain signal.  
* **Model Evolution (MoE vs. NN):** Early versions of mlgw utilized a **Mixture of Experts (MoE)** model for regression. However, the snippets indicate a transition in later versions (specifically Version 3\) to **Neural Networks**.6 The reference paper (2402.06587) can be seen as the "Higher Order Mode" extension of mlgw.

#### **Performance and Scope**

* **Training Data:** The model was trained on $\\mathcal{O}(10^3)$ waveforms from **TEOBResumS** and **SEOBNRv4**.5  
* **Accuracy:** It achieved mismatches at the $10^{-3}$ level.  
* **Speed:** It demonstrated a speedup factor of 10–50x for single waveforms and orders of magnitude more for batches.5  
* **Parameter Space:** Mass ratios $q \\in $ and spins $s \\in \[-0.8, 0.95\]$.

#### **Insight: The "Closed Form" Advantage**

One unique insight from the mlgw paper is that the model provides a "closed form expression" for the waveform and its gradient with respect to orbital parameters.5 Because the neural network is a differentiable chain of operations, and the PCA reconstruction is linear, the derivative of the waveform $\\partial h / \\partial \\theta$ can be computed via automatic differentiation (backpropagation) instantly. This is a massive advantage for Hamiltonian Monte Carlo (HMC) sampling methods in Bayesian inference, which require gradients—something traditional EOB codes cannot easily provide.

### **3.2 mlgw\_bns: Extension to Matter Effects**

The technique proved robust enough to be extended beyond vacuum Black Hole binaries to **Binary Neutron Stars (BNS)**. The package mlgw\_bns 8 adapts the architecture to include tidal deformability parameters $\\Lambda\_1, \\Lambda\_2$.

#### **Technical Adaptation**

* **Complexity:** BNS waveforms contain tidal effects that manifest primarily in the high-frequency phase evolution near the merger.  
* **Data Efficiency:** The snippets reveal that mlgw\_bns can reconstruct waveforms with mismatches lower than $10^{-4}$ using as few as **1,000 training waveforms**.8 This highlights the extreme data efficiency of the PCA-based approach; because the PCA basis captures the dominant physical features (the "shape" of a chirp), the neural network only needs to learn the coefficients, which requires far less data than training a model to generate time-series point-by-point.

### **3.3 Comparative Table: Grimbergen et al. vs. mlgw**

| Feature | Grimbergen et al. (2402.06587) | mlgw (Schmidt et al.) |
| :---- | :---- | :---- |
| **Primary Goal** | **Higher Order Modes (HOM)** | Fast generation of dominant (2,2) mode |
| **Approximant** | SEOBNRv4HM | TEOBResumS / SEOBNRv4 |
| **Regression** | Ensemble of Neural Networks | Mixture of Experts (v1/v2) $\\rightarrow$ NN (v3) |
| **Dim. Reduction** | PCA (Amp/Phase) | PCA (Amp/Phase) |
| **Modes Modeled** | Multiple $\\ell, m$ modes | Initially (2,2), extended later |
| **Speedup** | $\\sim 100x$ (single) | 10-50x (single) |

## **4\. NRSurNN: The "Transfer Learning" Revolution**

While Grimbergen et al. focused on emulating *approximants* (models that are already approximations of General Relativity), a more ambitious class of papers applies the same PCA+NN technique to emulate **Numerical Relativity (NR)** simulations directly. NR is the "ground truth," but simulations are scarce.

The paper **"NRSurNN3dq4: A Deep Learning Powered Numerical Relativity Surrogate for Binary Black Hole Waveforms"** (Freitas et al., arXiv:2412.06946) 10 represents the state-of-the-art in this domain.

### **4.1 The Data Scarcity Bottleneck**

Training a neural network typically requires tens of thousands of samples ($10^5$ in Grimbergen et al.). However, the available catalog of high-quality NR waveforms (e.g., from the SXS collaboration) numbers in the low thousands. Direct application of the "Grimbergen Technique" would lead to massive overfitting.

### **4.2 The Solution: Transfer Learning**

Freitas et al. solved this by modifying the training pipeline while keeping the core architecture (PCA+NN) intact.

1. Pre-training on Approximants:  
   They first generate a massive dataset ($10^6$ waveforms) using an existing hybrid surrogate model (NRSur7dq4 or similar).10 They train the Neural Network to learn the mapping $\\theta \\rightarrow c\_k^{approx}$.  
   * *Insight:* This teaches the network the "topology" of the parameter space—how mass ratio and spin generally affect the waveform coefficients.  
2. Fine-tuning on NR Data:  
   They then take this pre-trained network and retrain it on the small set of actual NR simulations (381 waveforms).12 They use a lower learning rate ($1 \\times 10^{-4}$) to gently adjust the weights.  
   * *Insight:* The network "corrects" the approximations of the surrogate to match the exact NR truth. This allows the model to achieve NR-level accuracy with only hundreds of true training samples.

### **4.3 Architectural Specifics**

The snippets provide detailed insights into the NRSurNN architecture, allowing for a precise comparison with 2402.06587:

* **Network Structure:** A Multi-Layer Perceptron (MLP) with 3 hidden layers containing **64, 512, and 1024 neurons** respectively.12  
* **Activation:** ReLU (Rectified Linear Unit).  
* Loss Function: A composite loss function is used:

  $$L \= L\_{MAE} \+ \\log(L\_{Mismatch})$$

  It combines the Mean Absolute Error of the PCA coefficients ($L\_{MAE}$) with a "physics-informed" term ($L\_{Mismatch}$) that measures the overlap of the reconstructed waveform.12 This ensures the network prioritizes coefficients that physically matter most for detection.  
* **Performance:** The model achieves mismatches $\< 10^{-3}$ and generates millions of waveforms in 0.1 seconds on a GPU.10

This paper is a critical "same kind of technique" result because it validates that the PCA+NN architecture is not just a way to speed up approximants, but a viable path to creating **new, higher-accuracy models** that were previously impossible to build due to data constraints.

## **5\. SEOBNRE\_AI: Handling the Eccentricity Challenge**

The reference paper 2402.06587 is limited to *quasi-circular* binaries. A parallel body of work applies the same PCA+NN technique to *eccentric* binaries (where the orbit is elliptical). The primary paper here is **"SEOBNRE\_AIq5e2: A Deep Learning Powered Surrogate for Eccentric Binary Black Hole Waveforms"**.13

### **5.1 The Variable Length Problem**

A major technical hurdle in adapting the "Grimbergen Technique" to eccentricity is the variable duration of the signals.

* In quasi-circular mergers, the time-to-merger is a relatively simple function of the chirp mass.  
* In eccentric mergers, the duration depends complexly on the initial eccentricity $e\_0$ and the evolution of the orbit.  
* **PCA Constraint:** Principal Component Analysis requires the input data (the training waveforms) to be vectors of fixed dimension $D$. If one waveform has 10,000 points and another has 50,000, standard PCA fails.

### **5.2 The SEOBNRE\_AI Solution: Adaptive Resampling**

The authors of SEOBNRE\_AIq5e2 introduce a novel preprocessing step to preserve the PCA+NN pipeline:

1. **Resampling:** All training waveforms, regardless of physical duration, are interpolated onto a fixed grid of $N=1024$ points.14 This forces the data into a fixed-dimensional vector space suitable for PCA/NN.  
2. **Dual-Network Architecture:**  
   * **Network A (Waveform Model):** Predicts the amplitude and phase on the *fixed* 1024-point grid.  
   * **Network B (Length Model):** A separate MLP is trained to predict the *physical duration* $T(\\theta)$ of the waveform.14  
3. **Reconstruction:** During inference, Network A generates the shape, Network B generates the duration, and the result is interpolated back onto the physical time axis.

### **5.3 Performance and Implications**

This modification allows the model to handle mass ratios up to $q=5$, eccentricities $e \\le 0.2$, and spins $|\\chi\_z| \\le 0.6$.14

* **Speed:** 4.3 ms per waveform generation.16  
* **Accuracy:** Mean mismatch of $1.02 \\times 10^{-3}$.

This paper demonstrates the flexibility of the PCA+NN technique. Even when the fundamental assumption of the data structure (fixed length) is violated, the architecture can be adapted via intelligent preprocessing and auxiliary networks.

## **6\. Emerging Architectural Divergences**

While the PCA+NN backbone (as used in 2402.06587, mlgw, NRSurNN, and SEOBNRE\_AI) is dominant, several papers in the provided research material explore alternative architectures that aim to achieve the same goal (fast, accurate waveform generation) but through different deep learning mechanisms.

### **6.1 Transformer Architectures (arXiv:2409.03833)**

The paper *Higher-order gravitational wave modes... using transformer architectures* 17 challenges the PCA approach.

* **The Critique of PCA:** PCA is a global linear reduction. It assumes that the "basis functions" are constant across the parameter space. However, the morphology of waveforms (especially ringdown and merger) changes non-linearly.  
* **The Transformer Approach:** Instead of predicting global coefficients, this model treats the waveform generation as a **Sequence-to-Sequence** task, similar to Natural Language Processing (NLP). The model uses **Self-Attention** mechanisms to predict the waveform evolution step-by-step or in latent patches.  
* **Results:** The transformer model showed the ability to generalize well to out-of-distribution systems (e.g., mass ratios up to $q=15$ when trained on lower $q$), a capability that PCA-based models often struggle with due to the rigidity of the basis.17

### **6.2 Fourier Analysis Networks (FAN)**

The paper *Compact binary systems waveform generation with a generative pretrained transformer* 18 mentions the use of **Fourier Analysis Networks**.

* **Mechanism:** Standard MLPs use activation functions like ReLU or Tanh. FANs use trigonometric activation functions ($\\sin, \\cos$).  
* **Motivation:** This embeds the periodic nature of gravitational waves directly into the neurons of the network. This combats "spectral bias"—the tendency of neural networks to learn low-frequency features faster than high-frequency ones.  
* **Performance:** The authors claim this mitigates the need for extremely deep networks to capture the oscillatory phase evolution, offering a potential efficiency gain over the standard MLP used in Grimbergen et al..18

### **6.3 Conditional Variational Autoencoders (CVAE) for Ringdown**

Another variation is the use of CVAEs, described in *Conditional Variational Autoencoder... for accelerated ringdown parameter estimation*.19

* **Difference:** The "Grimbergen Technique" is a *deterministic surrogate* (Input Parameters $\\rightarrow$ One Waveform). A CVAE is a *probabilistic generative model* (Input Parameters \+ Latent Noise $\\rightarrow$ Distribution of Waveforms).  
* **Application:** This is particularly powerful for analyzing the **Ringdown** phase (the post-merger signal), where uncertainties in the start time and mode content are high. The CVAE can model the *posterior distribution* of the ringdown parameters directly, effectively bypassing the likelihood evaluation step entirely in some pipelines.19

## **7\. Comparative Analysis of Identified Methodologies**

The following table synthesizes the technical specifications of the key papers identified, contrasting them with the reference work (2402.06587).

| Feature | Grimbergen et al. (2402.06587) | mlgw (Schmidt et al.) | NRSurNN3dq4 (Freitas et al.) | SEOBNRE\_AI | Transformer Models |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Technique Class** | **Time-Domain PCA \+ NN** | **Time-Domain PCA \+ NN** | **Time-Domain PCA \+ NN \+ Transfer** | **PCA \+ NN \+ Resampling** | **Sequence-to-Sequence** |
| **Dimensionality Reduction** | PCA (Amp/Phase separated) | PCA (Amp/Phase separated) | PCA (Amp/Phase separated) | Fixed-grid Resampling | Latent Embeddings |
| **Regression Model** | Ensemble of MLPs | MoE (early) $\\rightarrow$ NN (v3) | Pre-trained MLP (Transfer Learning) | Dual MLP (Shape \+ Length) | Transformer (Attention) |
| **Training Data** | SEOBNRv4HM (Approximant) | TEOBResumS / SEOBNRv4 | **Numerical Relativity (SXS)** | SEOBNRE (Eccentric) | NR / Approximant |
| **Physics Focus** | **Higher Order Modes (HOM)** | Speed / BNS extension | **NR Accuracy** / High Fidelity | **Eccentricity** | Sequence Modeling / Generalization |
| **Speedup** | $\\sim 100x$ | 10-50x | $\\sim 1000x$ (vs NR) | 4.3 ms / waveform | Variable |
| **Key Innovation** | Efficient HOM modeling | First robust python package | Solving NR data scarcity | Handling variable length | Capturing non-linear dependencies |

## **8\. Quantitative Performance and Hardware Acceleration**

A crucial aspect of these papers is not just the architecture, but the realized computational gains. The "same kind of technique" implies a shared goal: drastic acceleration of Bayesian inference.

### **8.1 The GPU Advantage**

The snippet regarding **Gravitational-wave surrogate models powered by artificial neural networks** (ANN-Sur) 20 provides concrete metrics on hardware scaling.

* **CPU vs. GPU:** A single waveform generation might take \~1-2 ms on a CPU. On a GPU, this drops to **0.0016 ms per waveform** when generated in batches of $10^3$ to $10^4$.20  
* **Throughput:** This translates to generating millions of waveforms per second.  
* **Implication:** This enables "Global Fitting" strategies where an entire catalog of events could be re-analyzed in minutes rather than months.

### **8.2 Accuracy vs. Speed Trade-off**

The NRSurNN3dq4 paper 10 highlights a critical trade-off. While the Neural Network is slightly less accurate than a pure interpolation-based surrogate (like standard spline interpolation) within the training set, it is **orders of magnitude faster** and has a smaller memory footprint (KB vs GB). For parameter estimation, where sampling noise often dominates, the $10^{-3}$ mismatch floor of these NN models is generally acceptable, making the speed gain pure profit for the analyst.

## **9\. Insights and Future Outlook**

The convergence of multiple research groups (Utrecht, Pisa, SXS Collaboration, etc.) on the **PCA \+ Neural Network** architecture indicates that this is a stable, mature "design pattern" in gravitational wave physics.

### **9.1 The "Manifold Hypothesis" in GWs**

The success of these papers validates the "Manifold Hypothesis" for gravitational waves: that the set of all physically valid waveforms lies on a low-dimensional manifold embedded in the high-dimensional space of time series. PCA finds the tangent plane to this manifold; the Neural Network parameterizes the coordinates on this plane.

* *Insight:* The fact that linear PCA works so well (as seen in Grimbergen and mlgw) suggests that, for non-precessing binaries, the waveform manifold is surprisingly flat. The struggle to extend this to **precession** (where the manifold becomes highly curved and higher-dimensional) is the current frontier, as hinted at in the NRSurNN discussions on data scarcity.4

### **9.2 Democratization of NR Data**

The NRSurNN transfer learning technique 10 is perhaps the most impactful "second-order" insight. It suggests that we do not need millions of NR simulations. We only need enough NR simulations to *calibrate* the biases of approximate models. This could significantly reduce the cost of future NR campaigns, guiding them to simulate only the points in parameter space where the neural network uncertainty is highest (Active Learning).

### **9.3 Integration into bilby and Production Pipelines**

The snippets mention the implementation of these models into bilby 10 and their use in re-analyzing GWTC-1.5 This marks the transition of the "Grimbergen Technique" from experimental code to **critical infrastructure**. Future detection pipelines for 3G detectors (Einstein Telescope) will likely rely exclusively on such surrogates for real-time alerts, as traditional codes will be too slow to process the overlapping signals expected in those detectors.1

## **10\. Conclusion**

The user's query identifying paper 2402.06587 (*Grimbergen et al.*) points to a specific and highly effective lineage of research in gravitational wave modeling. The "same kind of technique" is defined by the **decomposition of time-domain waveforms into amplitude/phase, the reduction of these components via PCA, and the mapping of physical parameters to PCA coefficients using Deep Neural Networks.**

This report has identified the following key papers and tools that share this DNA:

1. **mlgw (Schmidt et al.):** The foundational work and direct precursor.  
2. **NRSurNN3dq4 (Freitas et al.):** The high-accuracy evolution applying transfer learning to Numerical Relativity.  
3. **SEOBNRE\_AI:** The adaptive evolution handling eccentric orbits.  
4. **mlgw\_bns:** The extension to matter-containing binary neutron stars.

Collectively, these works demonstrate that machine learning in GW physics has moved beyond simple classification tasks to becoming the primary engine for high-fidelity, high-speed waveform generation, a capability that is prerequisite for the science goals of the next decade.

#### **Works cited**

1. \[2402.06587\] Generating Higher Order Modes from Binary Black Hole mergers with Machine Learning \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2402.06587](https://arxiv.org/abs/2402.06587)  
2. arXiv:2402.06587v2 \[gr-qc\] 26 Apr 2024, accessed December 22, 2025, [https://arxiv.org/pdf/2402.06587](https://arxiv.org/pdf/2402.06587)  
3. Frequency-domain reduced-order model of aligned-spin effective-one-body waveforms with higher-order modes \- Semantic Scholar, accessed December 22, 2025, [https://www.semanticscholar.org/paper/Frequency-domain-reduced-order-model-of-waveforms-Cotesta-Marsat/9a016ae73e3c9db8170127df26a1cf92e6e13751](https://www.semanticscholar.org/paper/Frequency-domain-reduced-order-model-of-waveforms-Cotesta-Marsat/9a016ae73e3c9db8170127df26a1cf92e6e13751)  
4. Optimizing Neural Network Surrogate Models: Application to Black Hole Merger Remnants | Request PDF \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/publication/388459298\_Optimizing\_Neural\_Network\_Surrogate\_Models\_Application\_to\_Black\_Hole\_Merger\_Remnants](https://www.researchgate.net/publication/388459298_Optimizing_Neural_Network_Surrogate_Models_Application_to_Black_Hole_Merger_Remnants)  
5. Machine learning gravitational waves from binary black hole mergers \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/publication/349618954\_Machine\_learning\_gravitational\_waves\_from\_binary\_black\_hole\_mergers](https://www.researchgate.net/publication/349618954_Machine_learning_gravitational_waves_from_binary_black_hole_mergers)  
6. stefanoschmidt1995/MLGW: A Machine Learning model for generating gravitational waves, accessed December 22, 2025, [https://github.com/stefanoschmidt1995/MLGW](https://github.com/stefanoschmidt1995/MLGW)  
7. mlgw \- PyPI, accessed December 22, 2025, [https://pypi.org/project/mlgw/](https://pypi.org/project/mlgw/)  
8. jacopok/mlgw\_bns: Accelerating gravitational wave template generation with machine learning. \- GitHub, accessed December 22, 2025, [https://github.com/jacopok/mlgw\_bns/](https://github.com/jacopok/mlgw_bns/)  
9. cv | Jacopo Tissino \- GitHub Pages, accessed December 22, 2025, [https://jacopok.github.io/cv/](https://jacopok.github.io/cv/)  
10. NRSurNN3dq4: A Deep Learning Powered Numerical Relativity Surrogate for Binary Black Hole Waveforms \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2412.06946v1](https://arxiv.org/html/2412.06946v1)  
11. NRSurNN3dq4: A Deep Learning Powered Numerical Relativity Surrogate for Binary Black Hole Waveforms | Request PDF \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/publication/386946347\_NRSurNN3dq4\_A\_Deep\_Learning\_Powered\_Numerical\_Relativity\_Surrogate\_for\_Binary\_Black\_Hole\_Waveforms](https://www.researchgate.net/publication/386946347_NRSurNN3dq4_A_Deep_Learning_Powered_Numerical_Relativity_Surrogate_for_Binary_Black_Hole_Waveforms)  
12. NRSurNN3dq4 \- Agenda INFN, accessed December 22, 2025, [https://agenda.infn.it/event/40538/contributions/243020/attachments/126228/186302/GRASS\_Osvaldo.pdf](https://agenda.infn.it/event/40538/contributions/243020/attachments/126228/186302/GRASS_Osvaldo.pdf)  
13. Fc coins Buyfc26coins.com has the perfect answer : What records should I keep when buying FC 26 Coins with crypto?.4tEl | Qeios, accessed December 22, 2025, [https://www.qeios.com/search?q=Fc%20coins%20Buyfc26coins.com%20has%20the%20perfect%20answer%20%3A%20What%20records%20should%20I%20keep%20when%20buying%20FC%2026%20Coins%20with%20crypto%3F.4tEl\&page=56](https://www.qeios.com/search?q=Fc+coins+Buyfc26coins.com+has+the+perfect+answer+:+What+records+should+I+keep+when+buying+FC+26+Coins+with+crypto?.4tEl&page=56)  
14. Rapid Eccentric Spin-Aligned Binary Black Hole Waveform Generation Based on Deep Learning \- Qeios, accessed December 22, 2025, [https://www.qeios.com/read/MFL8ZK](https://www.qeios.com/read/MFL8ZK)  
15. \[2411.14893\] Rapid eccentric spin-aligned binary black hole waveform generation based on deep learning \- arXiv, accessed December 22, 2025, [https://arxiv.org/abs/2411.14893](https://arxiv.org/abs/2411.14893)  
16. Rapid eccentric spin-aligned binary black hole waveform generation based on deep learning \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2411.14893v1](https://arxiv.org/html/2411.14893v1)  
17. Sequence modeling of higher-order wave modes of quasi-circular, spinning, non-precessing binary black hole mergers \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2409.03833v2](https://arxiv.org/html/2409.03833v2)  
18. Compact binary systems waveform generation with a generative pretrained transformer, accessed December 22, 2025, [https://www.researchgate.net/publication/379736341\_Compact\_binary\_systems\_waveform\_generation\_with\_a\_generative\_pretrained\_transformer](https://www.researchgate.net/publication/379736341_Compact_binary_systems_waveform_generation_with_a_generative_pretrained_transformer)  
19. Black Hole Spectroscopy with Conditional Variational Autoencoder \- arXiv, accessed December 22, 2025, [https://arxiv.org/html/2506.17618v1](https://arxiv.org/html/2506.17618v1)  
20. Gravitational-wave surrogate models powered by artificial neural networks \- ResearchGate, accessed December 22, 2025, [https://www.researchgate.net/publication/349983994\_Gravitational-wave\_surrogate\_models\_powered\_by\_artificial\_neural\_networks](https://www.researchgate.net/publication/349983994_Gravitational-wave_surrogate_models_powered_by_artificial_neural_networks)
