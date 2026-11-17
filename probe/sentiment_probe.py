"""
Linear probes on GPT-2: does the model internally represent sentiment, and where?

What this script does
  1. Runs GPT-2 on movie-review sentences and saves the hidden state of the
     last token at every layer.
  2. Trains one logistic-regression "probe" per layer to predict sentiment.
  3. Tests each probe on (a) held-out data from the same dataset,
     (b) data from a different dataset (distribution shift), and
     (c) a control: probes trained on shuffled labels (should be ~50%).
  4. Shows where the best probe fails (confident mistakes + tricky sentences).

Run in Colab or locally:
    pip install torch transformers datasets scikit-learn matplotlib
    python sentiment_probe.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MODEL = "gpt2"      # try "distilgpt2" for speed, or "gpt2-medium" for more layers
N_TRAIN = 2000      # training sentences
N_TEST = 800        # test sentences per dataset
BATCH = 32
SEED = 0

rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token            # GPT-2 has no pad token by default
model = AutoModel.from_pretrained(MODEL, output_hidden_states=True).to(device).eval()


@torch.no_grad()
def get_acts(texts):
    """Return array [n_texts, n_layers+1, hidden_dim]: last-token state per layer."""
    chunks = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=64).to(device)
        hidden = model(**enc).hidden_states            # tuple of [B, T, D]
        last = enc["attention_mask"].sum(1) - 1        # index of last real token
        idx = torch.arange(len(batch), device=device)
        chunks.append(torch.stack([h[idx, last] for h in hidden], dim=1).cpu())
    return torch.cat(chunks).numpy()


def sample(split, n):
    idx = rng.choice(len(split), size=min(n, len(split)), replace=False)
    sub = split.select(idx.tolist())
    text_key = "sentence" if "sentence" in sub.column_names else "text"
    return list(sub[text_key]), np.array(sub["label"])


# ---- Data: SST-2 for training/testing, Rotten Tomatoes as a shifted test set ----
sst2 = load_dataset("nyu-mll/glue", "sst2")
rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")

X_tr_txt, y_tr = sample(sst2["train"], N_TRAIN)
X_te_txt, y_te = sample(sst2["validation"], N_TEST)
X_ood_txt, y_ood = sample(rt["test"], N_TEST)

print("Extracting activations...")
A_tr, A_te, A_ood = get_acts(X_tr_txt), get_acts(X_te_txt), get_acts(X_ood_txt)
n_layers = A_tr.shape[1]


def make_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))


# ---- Train one probe per layer ----
acc_in, acc_ood, acc_ctrl = [], [], []
y_shuffled = rng.permutation(y_tr)       # control: labels with no real signal

for layer in range(n_layers):
    probe = make_probe().fit(A_tr[:, layer], y_tr)
    acc_in.append(probe.score(A_te[:, layer], y_te))
    acc_ood.append(probe.score(A_ood[:, layer], y_ood))

    ctrl = make_probe().fit(A_tr[:, layer], y_shuffled)
    acc_ctrl.append(ctrl.score(A_te[:, layer], y_te))
    print(f"layer {layer:2d} | same-data {acc_in[-1]:.3f} | "
          f"shifted-data {acc_ood[-1]:.3f} | shuffled-label control {acc_ctrl[-1]:.3f}")

# ---- Plot ----
plt.figure(figsize=(8, 5))
plt.plot(acc_in, marker="o", label="Test (SST-2, same distribution)")
plt.plot(acc_ood, marker="o", label="Test (Rotten Tomatoes, shifted)")
plt.plot(acc_ctrl, marker="o", linestyle="--", label="Control: shuffled labels")
plt.axhline(0.5, color="gray", linewidth=0.8)
plt.xlabel("Layer (0 = embeddings)")
plt.ylabel("Probe accuracy")
plt.title(f"Where is sentiment linearly readable in {MODEL}?")
plt.legend()
plt.tight_layout()
plt.savefig("probe_accuracy_by_layer.png", dpi=150)
print("Saved probe_accuracy_by_layer.png")

# ---- Failure analysis at the best layer ----
best = int(np.argmax(acc_in))
print(f"\nBest layer on same-distribution data: {best} ({acc_in[best]:.3f})")
probe = make_probe().fit(A_tr[:, best], y_tr)

probs = probe.predict_proba(A_te[:, best])[:, 1]
pred = (probs > 0.5).astype(int)
wrong = np.where(pred != y_te)[0]
wrong = wrong[np.argsort(-np.abs(probs[wrong] - 0.5))]    # most confident mistakes first

print("\nMost confident mistakes (label 1 = positive):")
for i in wrong[:8]:
    print(f"  true={y_te[i]} P(pos)={probs[i]:.2f} | {X_te_txt[i]}")

tricky = [
    "I didn't hate it.",
    "Not bad at all.",
    "Oh great, another two hours of my life wasted.",
    "The acting was good, but I would never watch it again.",
    "I expected to hate it, and I loved it.",
    "This film is not good.",
]
p = probe.predict_proba(get_acts(tricky)[:, best])[:, 1]
print("\nTricky sentences (negation / sarcasm / mixed):")
for s, pr in zip(tricky, p):
    print(f"  P(pos)={pr:.2f} | {s}")
