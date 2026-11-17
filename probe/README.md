# Where does GPT-2 represent sentiment? A layer-by-layer linear probe study

## Question

At which layers of GPT-2 (small) can the sentiment of a movie-review sentence be
read out by a simple linear probe, and does that readout generalize to a
different dataset?

I care about this question because it gives some insight into representation and
allows tracing for the factors resbonsible for the accuracy increase by layers.


## Method

- **Model:** GPT-2 small (12 layers, 768 hidden dimensions), frozen.
- **Features:** the hidden state of the last token at every layer (layer 0 is the
  embedding output, layer 12 is the final output).
- **Probe:** logistic regression on standardized features (`C = 0.1`), one probe
  per layer.
- **Data:**
  - Train: 2,000 sentences from SST-2 (train split).
  - Test, same distribution: 800 sentences from SST-2 (validation split).
  - Test, shifted distribution: 800 sentences from Rotten Tomatoes (test split).
- **Control:** the same probes trained on randomly shuffled labels, which should
  score near 50%.
- **Failure analysis:** the most confident mistakes of the best layer, plus a
  handful of hand-written sentences with negation, sarcasm, and mixed sentiment.

All results below are from a single run with seed 0.

## Results

![Probe accuracy by layer](probe_accuracy_by_layer.png)

| Layer | Same data (SST-2) | Shifted data (Rotten Tomatoes) | Shuffled-label control |
|------:|------------------:|-------------------------------:|-----------------------:|
| 0  | 0.501 | 0.500 | 0.522 |
| 1  | 0.736 | 0.502 | 0.507 |
| 2  | 0.749 | 0.549 | 0.453 |
| 3  | 0.731 | 0.616 | 0.470 |
| 4  | 0.744 | 0.509 | 0.485 |
| 5  | 0.767 | 0.509 | 0.494 |
| 6  | 0.764 | 0.540 | 0.509 |
| 7  | 0.809 | 0.721 | 0.489 |
| 8  | 0.834 | 0.654 | 0.512 |
| 9  | **0.841** | **0.775** | 0.564 |
| 10 | 0.840 | 0.759 | 0.535 |
| 11 | 0.831 | 0.744 | 0.514 |
| 12 | 0.828 | 0.741 | 0.489 |

### What I observed

1. **Sentiment is most readable in the middle-to-late layers.** Accuracy on the
   same distribution rises from chance at layer 0 to about 84% at layers 9 and
   10, then dips slightly.
2. **Probes from early layers did not transfer.** Layers 1 to 6 reach 73 to 77%
   on SST-2 but only about 50 to 62% on Rotten Tomatoes. Layers 7 to 12 keep
   about 65 to 78%. One possible explanation is that early-layer probes rely on
   dataset-specific surface cues, but I have not tested this.
3. **The control behaves as expected.** Shuffled-label probes stay between 45%
   and 56%, close to chance.

### Failure analysis

- Several of the most confident mistakes involve sarcasm or idiom, for example
  "it has all the excitement of eating oatmeal" (predicted positive with
  probability 0.99).
- Some "mistakes" look like questionable labels. One sentence containing
  "a terrific climax, and some nice chills" is labeled negative.
- In a small hand-written test (6 sentences, so anecdotal), negation such as
  "This film is not good" was handled correctly, but the sarcastic "Oh great,
  another two hours of my life wasted" was predicted positive with probability
  1.00.
- The probe's probabilities are almost always near 0 or 1, so it is overconfident.

## Limitations

- **Single run, single seed.** The layer-to-layer differences in shifted-data
  accuracy (for example 0.721 at layer 7, 0.654 at layer 8, 0.775 at layer 9) may
  be noise. [TODO: rerun with several seeds and report mean and spread.]
- **No simple baseline yet.** I do not know how much of the roughly 84% a
  bag-of-words classifier would reach. [TODO: add a TF-IDF logistic regression
  baseline and report it.]
- **A probe reading a property does not show the model uses it.** These results
  say sentiment is linearly decodable, not that GPT-2 relies on that direction.
- **Last-token features only.** Other pooling choices (for example the mean over
  all tokens) might behave differently. [TODO: optional follow-up.]
- **One small model and two datasets.** Nothing here should be assumed to hold
  for larger models or other kinds of text.
- **Label noise** in SST-2 and Rotten Tomatoes limits the achievable accuracy.

## How to run

```bash
pip install torch transformers datasets scikit-learn matplotlib
python sentiment_probe.py
```

This prints per-layer accuracies and the failure analysis, and saves
`probe_accuracy_by_layer.png`. A GPU (for example free Colab) makes feature
extraction faster, but a laptop CPU also works for this size.

Note: recent versions of the Hugging Face libraries need dataset names with the
owner prefix, which the script already uses (`nyu-mll/glue` and
`cornell-movie-review-data/rotten_tomatoes`).
