# CISC3024 AI Assignment #2 — Class-Conditional VAE Bayesian Classifier

A generative Bayesian classifier for MNIST: one convolutional Variational
Autoencoder per digit class, classified by Bayes' decision rule
`argmax_c p(x | ω_c) P(ω_c)` with the per-class ELBO as the likelihood
surrogate.

**Test accuracy: 94.31%** (9431 / 10,000), trained on CPU in ~8 minutes.

Everything (algorithm search, code, experiments, report) was produced by an
AI assistant for CISC3024 Pattern Recognition, AI Assignment #2.

## Repository layout

```
├── src/
│   ├── train_cvae_bayes.py     # model + training + Bayes inference + plots
│   └── results/                # metrics, figures, model checkpoints
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python src/train_cvae_bayes.py            # defaults: 5 epochs, batch 128, latent 20
```

Outputs (in `src/results/`):

- `metrics.txt` / `metrics.json` — accuracy, priors, settings
- `training_curves.png` — per-class BCE / KL curves
- `confusion_matrix.png` — 10×10 test confusion matrix
- `generated_samples.png` — random samples from each class VAE
- `reconstructions.png` — every test image × every class VAE
- `vae_class{0..9}.pt` — trained checkpoints

## How classification works

1. Train VAE_c on class-c images only → approximates p(x | ω_c).
2. For a test image compute −ELBO_c(x) for all c.
3. Predict `argmin_c ( −ELBO_c(x) − log P(ω_c) )`, with P(ω_c) the empirical
   class frequency. The evidence p(x) is constant across classes and cancels.

## Reference

Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014.
