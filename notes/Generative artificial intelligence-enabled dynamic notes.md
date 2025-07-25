## 1. High-level Architecture

```
           ┌──────────────────────┐
           │  Rat fMRI Acquisition│
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Pre-processing &    │
           │  Dynamic FC graphs   │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │ Graph Auto-Encoder   │   (Generative branch)
           └──────────┬───────────┘
                      │ recon. loss
                      ▼
           ┌──────────────────────┐
           │ Spatio-Temporal      │
           │ Graph Transformer    │   (Encoder ϕ)
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │ Contrastive InfoMax  │   (MI estimator ξ)
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │  Connection-Scoring  │   (Detector ψ)
           └──────────┬───────────┘
                      ▼
           Addiction-related circuits
```

### Modules

| Symbol | Component | Purpose |
|--------|-----------|---------|
| *GAE*  | Graph auto-encoder | Learns distribution of *normal* (saline) networks; augments data and regularises the latent space. |
| ϕ      | Spatio-Temporal Graph Transformer (SGT) | Extracts joint spatial (**brain topology**) and temporal (**fMRI dynamics**) features. |
| ξ      | Contrastive estimator | Maximises mutual information (InfoNCE) between local representations and a global summary vector. |
| ψ      | Scoring network | Assigns an addiction-likelihood score to every edge; top-k scores constitute detected circuits. |

---

## 2. Computational Pipeline

### Step-by-step Flow

1. **Animal experiment & fMRI**  
   • 21-day nicotine self-administration → 2-day withdrawal → acute injection during scan.  
   • 3 groups: Saline (S), Low-dose (L), High-dose (H).

2. **Pre-processing → Dynamic Graphs**  
   *SPM8* + motion correction + atlas parcellation (150 ROIs) → time-resolved FC matrices  
   $G^{(t)}=(V,E^{(t)},X^{(t)})$, $t=1…T$ (T≈10).

3. **Generative Pre-training (GAE)**  
   • Train a graph auto-encoder on *saline* data only.  
   • Reconstruction objective  
     
     $$\min_{\theta_g}\;\;\sum_{t}\|A^{(t)}-\hat A^{(t)}\|_F^2$$
     
   • Latent prior learned here is reused to initialise ϕ, mitigating small-sample over-fitting.

4. **Encoding with SGT (ϕ)**  

   For each time slice:
   ```
   h_i^0 = X_i
   for l = 1 … L:
       Spatial self-attention:          h_i^l ← AttnS(h_i^{l−1}, N(i))
       Temporal self-attention:         h_i^l ← AttnT(h_i^l, h_i^{l−1,t±1})
       Feed-forward + residual & norm
   ```
   Output: Node embeddings $H^{(t)}$ and a global token $g^{(t)}$.

5. **Contrastive InfoMax (ξ)**  

   • Positive pair: $(g^{(t)}, H^{(t)})$ from same graph.  
   • Negatives: shuffled time slices or different subjects.  
   • InfoNCE loss  
     
     $$\mathcal L_{Info}=-\sum_{t}\log \frac{\exp\,ξ(g^{(t)},H^{(t)})}{\sum_{t',s}\exp\,ξ(g^{(t)},H^{(t')}_s)}$$

6. **Connection Scoring (ψ)**  

   For every edge $e_{ij}^{(t)}$:
   
   $$s_{ij}^{(t)} = ψ\big([h_i^{(t)}\;\|\,h_j^{(t)}\;\|\,g^{(t)}]\big)$$

   Training loss = pairwise ranking: addicted edges above saline edges.

7. **Circuit Extraction**  
   • Keep top 1% edges per subject/dose.  
   • Aggregate over time to obtain dynamic circuit maps.  

---

## 3. Mathematical Objective

Total loss  

$$\mathcal L = λ_1\mathcal L_{recon} + λ_2\mathcal L_{Info} + λ_3\mathcal L_{rank}$$

* λ₁–λ₃ balance generative fidelity, representation quality, and discriminative scoring.

---

## 4. Algorithm Highlights

1. **End-to-end Dynamic Analysis**  
   The entire workflow—from raw fMRI to edge-level saliency—runs inside a single differentiable graph/Transformer pipeline.

2. **Spatio-Temporal Graph Transformer**  
   • Dual self-attention captures **non-Euclidean topology** and **temporal dependencies** simultaneously.  
   • Removes need for separate GCN + RNN stacks.

3. **Generative Regularisation for Small Samples**  
   Graph auto-encoder pre-training on plentiful *control* data provides a structured latent prior, greatly stabilising training on scarce addicted samples.

4. **Contrastive Learning without Labels**  
   InfoNCE forces ϕ to preserve intrinsic dynamics, yielding transferable representations even before supervised fine-tuning.

5. **Edge-wise Interpretability**  
   Scoring network produces explicit probabilities for every functional connection; ranking → transparent circuit maps approved by neuroscientists.

6. **Dose-dependent Discovery**  
   Parallel detectors for S-L and S-H pairs reveal shared vs. unique pathways, enabling mechanistic insight into how nicotine dosage modulates circuitry.

---

## 5. Pseudocode Summary

```python
# training loop
for epoch in range(E):
    for batch in loader:                 # dynamic graphs
        A_t, X_t, label = batch

        # -------- Generative branch --------
        z = GAE.encoder(A_t, X_t)
        A_hat = GAE.decoder(z)
        L_recon = mse(A_hat, A_t)

        # -------- Representation branch --------
        H_t, g_t = SGT(X_t, A_t)         # encoder ϕ
        L_info = InfoNCE(g_t, H_t)       # estimator ξ

        # -------- Scoring branch --------
        scores = ScoreNet(H_t, g_t)      # detector ψ
        L_rank = pairwise_rank(scores, label)

        loss = λ1*L_recon + λ2*L_info + λ3*L_rank
        loss.backward()
        optimiser.step()
```

---

## 6. Practical Impacts

* **Research tool** — automates discovery of fast, transient nicotine circuits that traditional static FC analyses miss.  
* **Translational potential** — edge ranking offers concrete targets for neuromodulation or pharmacological intervention.  
* **Scalable framework** — architecture agnostic to species; can ingest human rs-fMRI or other neuro-imaging modalities with minimal change.

---

### In a nutshell
The paper introduces a **generative-contrastive, Transformer-based graph framework** that dynamically dissects nicotine-related brain circuitry from very limited fMRI data, delivering both **state-of-the-art detection accuracy** and **granular neuro-biological interpretability**.