# Conference Abstract Draft (Vera Rubin Community)

**Title:** From Baseline to 92% Accuracy: An Iterative Approach to Light Curve Classification for the Rubin Era

**Author:** Giuliana Barbieri (ML Engineer & Student)

**Abstract:**
Title: Real-time Classification of Astronomical Transients: A Scalable Pipeline for the LSST Era
Abstract:
The Vera C. Rubin Observatory will soon transform time-domain astronomy by generating 10 million alerts per night, a volume that exceeds human capacity and traditional processing. To navigate this "haystack" of data, high-speed automated architectures are no longer an option, but a necessity. We present DeepRubin-Explorer, a modular machine learning pipeline designed to classify four representative classes of known astronomical objects in real-time. By integrating the ALeRCE broker stream with a Temporal Convolutional Network (TCN), our system bypasses manual feature engineering to identify transient signatures directly from raw photometry.

To address the irregular cadence and heteroscedastic noise of the alert stream, we implement Gaussian Process Regression for robust light curve interpolation. The model achieves a peak validation accuracy of 92.16% across Quasars (QSO), Cepheids (CEP), and Supernovae (Type Ia and II), demonstrating reliable performance even in early-stage classification. Performance benchmarks show an end-to-end latency of 64.5 ms per alert, proving the system's viability for LSST-scale processing through current optimization of computational bottlenecks. Ultimately, this pipeline establishes a rigorous baseline for the known objects, providing the necessary framework to isolate anomalous signals from technosignature candidates to events critical for galactic habitability studies.
