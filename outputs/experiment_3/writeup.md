# Experiment 3 write-up (draft)

The KAN-ODE vector field takes the 20 latent state coordinates plus scaled time t/48. It never sees raw ICU variables: the coordinates are outputs of a learned encoder, so a name like latent_state_7 has no clinical meaning by itself. Attribution therefore shows which latent directions drive the dynamics, not which vital sign matters.

I probed 200 states (a seeded subsample of 7,200 held-out patient-time pairs, with 12 times drawn uniformly from [0, 48] h). The two highest-scoring inputs were latent_state_15 (score 3.93) and latent_state_18 (score 2.28). 55.7% of probed states fell outside the spline grid [-1.00, 1.00], where splines fade out and the SiLU base branch dominates, so edge shapes there should be read cautiously. Before pruning the network had 861 active edges; after pruning, 861 remained. Symbolic fitting with the library x, x^2, exp, sin, tanh and abs produced closed forms (saved in symbolic_formulas.txt).

The MLP-ODE has no comparable view: its tanh layers mix all inputs, so there is no per-edge function to plot. The KAN's advantage is exactly these one-dimensional curves, but only in latent coordinates. The LTC/CfC model is interpretable differently: each neuron carries a learned, input-dependent time constant that shows how quickly units respond to irregular gaps, yet it also acts on latent units. Neither model maps directly to raw variables, so clinical claims would need an observation-level attribution.

Check the saved figures and edit any sentence they do not support before submitting.
