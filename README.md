# DeepRubin-Explorer 🌌
### Real-time Transient Classification & Astrobiological Target Selection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Data: ALeRCE Broker](https://img.shields.io/badge/Data-ALeRCE%20Broker-orange)](https://alerce.online/)

## 🔭 Overview
This repository implements a Machine Learning pipeline designed for the **Vera C. Rubin Observatory's LSST** era. The goal is to move beyond static batch processing by implementing real-time classification of astronomical transients (SNe, AGNs, Variables) using streaming data from the **ALeRCE broker**.

Inspired by recent research in multiscale astrobiology (e.g., Ćiprijanović et al.), this project explores how high-cadence photometry can be used to identify anomalous signals that may warrant follow-up observations.

## 📡 Scientific Motivation
How do we find life in a haystack of 10 million alerts per night? The **Vera C. Rubin Observatory (LSST)** will revolutionize our understanding of the dynamic universe, but its true power for astrobiology lies in **Anomaly Detection**. 

This project implements a high-performance Machine Learning pipeline to classify known astronomical transients (Supernovae, Variable Stars, AGNs). By mastering the "Expected Universe," we enable the identification of the **"Unexpected"**:
1. **Technosignature Candidates:** Signals that deviate from known physical models.
2. **Galactic Habitability:** Mapping high-energy events (SNe) that influence the chemical enrichment and sterilization risks of planetary systems.
3. **Interstellar Objects:** Identifying non-periodic transients that could be interstellar scouts or anomalous bolides.

## 🧠 ML Engineering Challenges
Transitioning from industrial ML to Astrophysics requires addressing domain-specific constraints:
* **Irregular Sampling:** Handling non-equidistant time series (cadence-dependent data).
* **Heteroscedastic Noise:** Integrating measurement uncertainties ($\sigma$) directly into the loss function.
* **Domain Shift:** Training on synthetic data (ELAsTiCC) and deploying on real survey streams (ZTF/Rubin).

## 🛠️ Architecture
The project is structured following clean code principles for scientific reproducibility:
* `ingestion/`: API wrappers for ALeRCE and ZTF alert streams.
* `preprocessing/`: Gaussian Process (GP) interpolation and feature extraction.
* `models/`: PyTorch implementations of Time-Series Transformers and RNNs.
* `notebooks/`: Exploratory Data Analysis (EDA) and astrophysical validation.

## 📊 Data Source
Currently utilizing the **Zwicky Transient Facility (ZTF)** alert stream via the **ALeRCE Client**, serving as a high-fidelity precursor to the upcoming LSST data release.

## 📈 Roadmap
- [x] Data ingestion pipeline via ALeRCE API.
- [ ] Exploratory Data Analysis of SN Ia vs. SN II light curves.
- [ ] Implementation of a Deep Learning classifier (Temporal Convolutional Networks).
- [ ] Uncertainty estimation using Bayesian Neural Networks.

---

## 📈 Data Visualization

The project currently explores real-time astronomical transients. Below is an example of a **Type Ia Supernova (SNIa)** light curve (Object: **ZTF18adoojej**) retrieved from the ALeRCE broker. 

![Light Curve Sample](assets/light_curve_sample.png)

> **Scientific Note:** Notice the irregular gaps between observations and the characteristic brightness decay. These gaps represent the "missing data challenge" that we aim to solve using Gaussian Processes, as suggested by modern astrophysical deep learning research.

---
**Author:** Giuliana Barbieri — *ML Engineer exploring the intersection of Big Data and Extragalactic Astrophysics.* 



