## 1. Architectural Overview

FGSAN (Feature-selected Graph Spatial Attention Network) is an end-to-end framework designed for two simultaneous goals:

* Graph-level prediction: distinguish nicotine-addicted (NA) from healthy control (HC) subjects.  
* Node-level interpretation: automatically isolate a compact subset of brain regions (biomarkers) that drive the decision.

The network contains three tightly coupled modules:

| Module | Role | Core ideas |
| ------ | ---- | ---------- |
| 1. Graph Spatial Attention Encoder (E) | Converts *dynamic* fMRI graphs into low-dimensional node embeddings. | Multi-head graph attention augmented with **explicit spatial encoding** derived from functional connectivity. |
| 2. Bayesian Feature Selector (S) | Learns a binary mask over node embeddings to keep only informative regions. | Relaxed Bernoulli sampling + KL divergence sparsity regularizer. |
| 3. MLP Classifier (C) | Aggregates the masked node features into a graph-level diagnosis. | Simple mean readout followed by fully-connected layers. |

## 2. Detailed Computational Pipeline

### 2.1 Pre-processing & Graph Construction

1. Resting-state fMRI of rat brains → 150 anatomical ROIs.  
2. Time-varying magnitude-squared coherence between ROI pairs → dynamic adjacency tensors  
   \( \{A^{(t)} \}_{t=1}^{T} \in \mathbb{R}^{150 \times 150}\).  
3. Raw BOLD signals of each ROI at each time frame → feature matrix  
   \(X \in \mathbb{R}^{150 \times D}\) (here *D* = length of time window).

### 2.2 Encoder: Graph Spatial Attention Layers

For layer `l`:

1. **Attention coefficient**
   \[
   \alpha_{ij}^{(l)} =
   \dfrac{\exp\big(\text{tanh}\big([h_i^{(l)}W^{(l)} \parallel h_j^{(l)}W^{(l)}]\!\cdot\!c^{(l)}
   + S_\psi(x_i,x_j)\big)\big)}
   {\sum_{k\in\mathcal{N}(i)}\exp(\cdot)}
   \]

   * \(h_i^{(l)}\) – node representation at layer *l*.  
   * \(S_\psi(x_i,x_j)\) – **learnable scalar** encoding the spatial relation of nodes *i* and *j* (distance, connectivity strength, etc.).  
   * This term lets attention look beyond feature similarity and exploit *where* the regions sit in the functional network.

2. **Feature propagation**
   \[
   h_i^{(l+1)} = \sigma\Big(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{(l)}\,h_j^{(l)}W^{(l)}\Big)
   \]

3. Three such layers stack to produce final node embeddings \(H = [\tilde h_1,\dots,\tilde h_N]\).

### 2.3 Bayesian Feature Selector

Objective: sample a binary mask \(B\in\{0,1\}^{N}\) that turns *off* unimportant ROIs while keeping gradients differentiable.

1. **Posterior approximation**  
   Learn logits \(z \in (0,1)^N\).  
2. **Gumbel-Sigmoid (Relaxed Bernoulli) sampling**
   \[
   b_i = \sigma\!\Big(\frac{\log z_i - \log(1-z_i)+\log u_i - \log(1-u_i)}{r}\Big),
   \quad u_i\sim \text{Uniform}(0,1)
   \]
   where `r` is a temperature hyper-parameter.
3. **Loss terms**

   * Classification BCE loss on masked features.  
   * KL divergence \( \text{KL}(\text{Ber}(z)\,\|\,\text{Ber}(s))\) to pull the posterior towards a sparse prior `s` (e.g., 0.5 or smaller).  

### 2.4 Graph-level Classification

1. **Readout / Pooling**  
   \[
   g = \sigma\!\Big(\frac{1}{N}\sum_{i=1}^N b_i \tilde h_i\Big)
   \]
2. **MLP** → probability of addiction \( \hat y \).

### 2.5 Training Loop

```
for each epoch:
    for each subject graph (X, {A(t)}, y):
        H  = Encoder(X, {A(t)})
        B  = SampleMask(z)          # differentiable
        g  = Readout(H, B)
        ŷ  = MLP(g)
        L  = BCE(ŷ, y) + λ·KL(Ber(z)||Ber(s))
        back-prop + Adam update
```

## 3. Key Innovations and Strengths

1. **Spatially-Aware Attention**  
   Embeds *positional* priors directly into the attention scores, allowing the model to respect neuro-anatomical constraints instead of solely relying on learned weights.

2. **Integrated Feature Selection**  
   Unlike post-hoc saliency maps, the Bayesian selector is *in-the-loop*: it shapes the embeddings during training, yields intrinsic sparsity, and outputs an unapologetically discrete biomarker set.

3. **Dual-Objective Optimization**  
   The KL + BCE coupling balances predictive power and parsimony, mitigating over-fitting on noisy, high-dimensional fMRI features.

4. **Dynamic Connectivity Utilization**  
   By feeding the full time series and dynamic adjacency sequence, FGSAN captures temporal variations often ignored by static functional networks.

5. **Interpretability with Performance**  
   – State-of-the-art accuracy (≈ 82 % vs. 70 %-78 % for baselines).  
   – Recovers five ROIs (Midbrain, Diagonal domain, etc.) consistent with prior neurobiological findings.

6. **Lightweight & Modular**  
   No 3-D convolutions or recurrent units; can be plugged into other graph encoders or combined with alternative pooling schemes.

## 4. Practical Implications

* **Clinical translation** – yields a small, interpretable panel of brain regions that clinicians can monitor or target.  
* **Generalization** – the selector can adapt to other neuro-psychiatric conditions with minor changes.  
* **Scalability** – amenable to multi-GPU batched training owing to simple GAT layers.

## 5. Summary

FGSAN seamlessly marries spatially-aware graph attention with Bayesian feature selection, achieving both high diagnostic accuracy and human-readable biomarker discovery from noisy, high-dimensional fMRI data. Its architecture embodies a principled trade-off between expressiveness, sparsity, and interpretability—qualities that make it a compelling blueprint for future graph-based neuroimaging studies.