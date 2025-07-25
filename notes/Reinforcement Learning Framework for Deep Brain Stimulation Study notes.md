## 1. Architectural Decomposition  

| Layer | Main Components | Role |
|-------|-----------------|------|
| Synthetic Brain Environment | • Neuronal ensemble models:<br>-- Bonhoeffer-van der Pol (FitzHugh-Nagumo) oscillators (regular & chaotic)<br>-- Hindmarsh-Rose bursting neurons<br>• Global (mean-field) coupling ε<br>• δ-pulse stimulation channel | Emulates pathological synchrony found in Parkinson-like states and exposes an OpenAI-Gym API. |
| Observation Interface | Last *M* = 250 samples of the mean field $X(t)$ | Compresses high-dimensional neuronal activity into a 1-D surrogate that is measurable in vivo (local field potential). |
| Action Interface | Scalar pulse $A(t)\in[-A_{\max},A_{\max}]$ applied to all neurons every Δ time-step | Represents the DBS stimulation amplitude; can be duty-cycled via a skip factor κ. |
| Reward Shaper | $R_t = -\big(X(t)-\langle X\rangle_{M}\big)^2 - \beta\|A(t)\|$ | Drives the ensemble toward its intrinsic equilibrium while penalising energy delivery. |
| RL Agent(s) | • PPO algorithm (Actor–Critic)<br>• Two-hidden-layer MLP (64–64)<br>• γ = 0.99 | Learns pulse policies that minimise synchrony. |
| Multi-Agent Extension | Primary PPO for large-amplitude regime + Secondary PPO for residual low-amplitude regime | Provides piece-wise control over highly non-linear response curves. |

## 2. End-to-End Algorithmic Flow  

```text
1. Initialise environment 𝔼 with chosen neuron model, N≈1000, coupling ε.
2. Initialise PPO policy π_θ and value network V_ϕ.
3. For each episode:
      a. Reset 𝔼 → obtain state s₀ = (X_{-249:0}).
      b. For t = 0 … T:
            i.   Agent draws action a_t ~ π_θ(s_t).
            ii.  With probability 1/κ deliver δ-pulse A(t)=a_t else A(t)=0.
            iii. 𝔼 integrates ODEs for Δ, returns new mean field X_{t+1}.
            iv.  Assemble next state s_{t+1}, compute reward r_t.
            v.  Store (s_t,a_t,r_t,s_{t+1}) in rollout buffer.
      c.  Perform K PPO updates on θ,ϕ using collected trajectories.
4. Optional: train a second PPO on trajectories where |X| is already small.
5. Freeze policy; deploy in inference loop for quantitative studies (ε-sweep, κ-sweep, noise tests).
```

## 3. Key Design Choices  

1. **Model-Agnostic Gym**  
   All neuronal ODE or map equations are wrapped behind a unified Gym interface; researchers can swap in more sophisticated biophysical models without touching the agent code.  

2. **Mean-Field State Compression**  
   Feeding only the scalar mean field and its short history mimics realistic sensing (LFP/ECoG) and keeps policy networks lightweight, facilitating on-device deployment.

3. **Energy-Aware Reward**  
   The $-\beta|A|$ term explicitly trades stimulation economy against desynchronisation, allowing direct exploration of clinical safety/efficacy frontiers.

4. **Multi-Resolution Control via Dual Agents**  
   Because neuronal PRCs are amplitude-dependent, a *cascade* of PPOs (coarse → fine) yields finer residual suppression than a single policy.

5. **Skip-Parameter κ**  
   Training with full-rate pulses but executing every κ-th action empirically maps the minimal pulse-rate that still maintains suppression, a knob directly translatable to battery life in implantables.

6. **Robustness Validation**  
   Systematically injecting Gaussian noise on both observations and actuation demonstrates graceful degradation and defines acceptable sensor/actuator tolerances.

## 4. Experimental Insights  

1. **Universality Across Regimes**  
   – Regular oscillations (ε=0.03), chaotic oscillations (ε=0.02) and bursting dynamics (Hindmarsh-Rose, ε=0.2) are all tamed with the *same* network architecture and reward.  
2. **Suppression Coefficient**  
   Defined as $S = \frac{\text{std}(X_\text{before})}{\text{std}(X_\text{after})}$; values up to **S ≈ 33** were achieved for strongly coupled ensembles.  
3. **Energy vs. Performance**  
   Five-fold pulse skipping (κ=5) preserved substantial suppression while reducing total delivered charge—critical for implant longevity and tissue safety.  
4. **Constant-Stimulus Baseline**  
   RL beats any constant current strategy by >6× in energy efficiency and avoids quenching individual neuron activity (a known side-effect of large DC biases).  
5. **Noise Tolerance**  
   – Observation noise: negligible impact until σₓ ≈ 0.1 std(X).  
   – Actuation noise: graceful decline; chaotic regime robust up to σₐ ≈ mean(|A|).  

## 5. Algorithmic Highlights & Contributions  

1. **First Open-Source RL Gym for DBS Research**  
   Enables apple-to-apple benchmarking of control strategies with clear metrics (S, $A_\text{total}$).  

2. **Model-Free Closed-Loop DBS Controller**  
   PPO learns directly from data without explicit knowledge of biophysical parameters, ideal for heterogeneous patient populations.  

3. **Energy-Optimal, Safety-Aware Stimulation**  
   Reward shaping enforces minimal-intrusion policies, a prerequisite for clinical translation.  

4. **Multi-Agent Cascade for Non-Linear Regimes**  
   Demonstrates that layering simple agents can outperform monolithic policies on stiff, highly non-linear systems.  

5. **Comprehensive Sensitivity Analysis**  
   Coupling-strength sweeps, duty-cycle sweeps, and noise injections collectively map the controllability landscape and practical deployment bounds.

## 6. Potential Extensions  

* Plug-in additional neuron models (Hodgkin–Huxley, multi-compartment).  
* Replace PPO with offline RL or model-based RL to cut on-patient training time.  
* Embed learned policies on low-power hardware (e.g., FPGA) for real DBS devices.  
* Expand multi-agent library to cover stimulus frequency, pulse width, and electrode configuration dimensions.

---

**In summary**, this work delivers a fully-featured, modular reinforcement learning toolkit that bridges computational neuroscience and adaptive neuromodulation. Its architecture consciously balances biological realism, engineering constraints, and machine-learning practicality, making it a strong foundation for future closed-loop DBS research and development.