# LSTM vs RNN: Working Memory Prediction Analysis

Comparing LSTM and RNN architectures to understand how gating mechanisms in neural networks relate to human working memory demands during language processing.

## Overview

This project investigates whether the difference between LSTM (with gating) and simple RNN (without gating) models captures aspects of human working memory load. By analyzing residuals between these models, we can test whether LSTM's memory mechanisms explain variance in established working memory predictors from psycholinguistics.

### Research Question

**Do LSTMs' gating mechanisms better capture working memory demands in human language processing compared to simple RNNs?**

If yes, the residual variance (LSTM - RNN) should correlate with psycholinguistic measures of working memory load.

## Technical Stack

- **Language:** R
- **Key Libraries:** `lme4`, `tidyverse`, `ggplot2`
- **Models Compared:**
  - LSTM (Long Short-Term Memory) - 15 training seeds
  - RNN with ReLU activation - 15 training seeds  
  - RNN with Tanh activation - 15 training seeds
- **Data:** Natural Stories Corpus + Working Memory predictors

## Working Memory Measures Tested

### DLT Family (Dependency Locality Theory)
- Integration cost of syntactic dependencies
- Variants: Subject/Complement/Verb-specific costs
- Tests structural complexity effects

### LC Family (Left-Corner Parsing)
- Embedding depth and memory load
- Constituent length and closure effects
- Tests hierarchical processing

### ACT-R
- Activation-based retrieval from memory
- Tests interference and decay effects

## Methodology

### 1. Model Training
```
15 LSTM models (different random seeds)
15 RNN-ReLU models (different random seeds)
15 RNN-Tanh models (different random seeds)
```

### 2. Residual Calculation
```r
# For each LSTM-RNN pair:
LSTM_residual = LSTM_surprisal ~ RNN_surprisal
```
This isolates what LSTM captures beyond simple RNN.

### 3. Working Memory Correlation
```r
# Test if LSTM residuals predict WM measures:
WM_measure ~ LSTM_residual
```

### 4. Statistical Testing
- Linear regression for each WM predictor × model pair
- Bonferroni correction within WM families
- Visualization via heatmap with significance outlining

## Key Results

The analysis produces:
- **Coefficient matrix:** Each cell = correlation between model residual and WM predictor
- **Significance testing:** Bonferroni-corrected p-values
- **Visualization:** Color-coded heatmap with outlined significant effects

## Code Structure

```r
# 1. Compute LSTM-RNN residuals
compute_residuals_LSTM_RNN(df)

# 2. Build coefficient table
build_wm_df(df, WM_predictors)

# 3. Apply corrections
# Bonferroni within DLT and LC families

# 4. Visualize
# Heatmap with significance markers
```

## Visualization

The main output is a publication-ready heatmap showing:
- **X-axis:** Model pairs (LSTM - RNN combinations)
- **Y-axis:** Working memory predictors
- **Color:** Regression coefficient (blue = negative, red = positive)
- **Black outline:** Statistically significant (p < 0.05, Bonferroni-corrected)

## Research Implications

### If LSTM residuals correlate with WM measures:
Suggests LSTM's gating mechanisms capture human-like memory constraints

### If no correlation:
Suggests gating ≠ working memory (at least not these specific mechanisms)

## Technical Details

### Multiple Seeds
- Averaging over 15 random initializations
- Controls for model-specific quirks
- More robust than single-model analysis

### Bonferroni Correction
- Separate correction for DLT family (9 measures)
- Separate correction for LC family (10 measures)
- Prevents false positives from multiple comparisons

### Model Comparison Strategy
```
LSTM vs ReLU-RNN  → Tests gating + activation
LSTM vs Tanh-RNN  → Tests gating (matched activation)
```

## Dataset

Uses the Natural Stories Corpus with:
- Word-level surprisal from neural language models
- Psycholinguistic working memory predictors from Shain et al. (2022)
- Natural reading materials (stories)
- Comprehensive linguistic annotations

## Applications

This methodology can be extended to:
- Compare other model architectures (Transformer vs LSTM)
- Test different cognitive predictors (attention, prediction error)
- Analyze other language corpora
- Validate computational models of cognition

## Research Context

This work bridges:
- **Computational linguistics:** Neural language models
- **Psycholinguistics:** Working memory theories (DLT, LC)
- **Cognitive science:** Memory constraints in language processing
- **AI interpretability:** What do model components actually learn?

## Related Work

- Gibson (1998, 2000): Dependency Locality Theory
- Lewis & Vasishth (2005): ACT-R model of sentence processing
- Shain et al. (2022): Working memory predictors in Natural Stories
- Futrell et al. (2021): Natural Stories corpus

## Future Directions

- Test Transformer attention patterns vs working memory
- Add semantic coherence measures
- Real-time prediction of reading difficulty
- Extend to other languages

## Code Availability

All code is well-commented and modular:
- Easy to adapt for new model types
- Easy to add new cognitive predictors
- Reusable visualization pipeline

---

**Author:** Yasemin Gokcen 
**Collaborators/Advisors:** Dr. Rachel Ryskin, Dr. David Noelle
**Affiliation:** PhD Candidate, Cognitive & Information Sciences, UC Merced  
**Contact:** ygokcen@ucmerced.edu

## References

- Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies
- Shain, C., et al. (2022). Working memory load in Natural Stories
- Futrell, R., et al. (2021). Natural Stories corpus
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory (LSTM)
