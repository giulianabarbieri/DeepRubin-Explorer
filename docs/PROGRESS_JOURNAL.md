# DeepRubin-Explorer: Scientific Progress Journal

This document tracks the iterative development of the DeepRubin-Explorer pipeline. For a scientist, the history of these improvements is more valuable than just the final number, as it shows the reasoning and the "lessons learned" during the research.

---

## 🔬 Scientific Context
The goal of this project is to classify astronomical transients (Supernovae, Quasars, etc.) in real-time. The main challenge is the **Class Imbalance** (some classes have fewer examples than others) and the **Irregular Cadence** of telescope data.

---

## 📝 Iteration Log

### Iteration 1: The Baseline (v1.0)
- **Goal**: Build a basic pipeline from data ingestion to classification.
- **Data**: 274 samples (91 QSO, 88 CEP, 66 SNIa, 29 SNII).
- **Architecture**: Standard Temporal Convolutional Network (TCN).
- **Loss Function**: Standard `CrossEntropyLoss`.
- **Observation**: The model achieved **67.3% Accuracy**. However, the confusion matrix showed that it failed to identify SNII (Type II Supernovae) because they were very rare in the training set.
- **Scientific Hypothesis**: The model is biased toward frequent classes. We need more data and a way to tell the model that rare classes are more important.

### Iteration 2: Scaling and Balancing (v2.0)
- **Goal**: Address class imbalance and improve general performance.
- **Action 1 (Data Expansion)**: We expanded the dataset to **764 samples** (400 QSO, 252 CEP, 77 SNIa, 35 SNII) using the ALeRCE broker more aggressively.
- **Action 2 (Weighted Cost-Function)**: Instead of a standard loss, we implemented a **Weighted CrossEntropyLoss**. We assigned higher weights only to the classes with fewer samples.
- **Action 3 (Automated Evaluation)**: We integrated automated confusion matrix generation into MLflow to see performance instantly.
- **Result**: The Peak Validation Accuracy jumped to **92.16%**.
- **Learning**: Combining more training examples with a cost-sensitive loss function is extremely effective for astronomical classification. The model became much better at distinguishing between Supernova types.

---

## 📈 Future Hypotheses
1. **Dynamic Windows**: 100 days might be too long for some transients. Reducing the window might help identify SNe earlier.
2. **Additional Features**: Including the $g-r$ color evolution as a separate channel might improve the classification of SNIa vs SNII.
