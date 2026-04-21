# Conference Abstract Draft (Vera Rubin Community)

**Title:** From Baseline to 92% Accuracy: An Iterative Approach to Light Curve Classification for the Rubin Era

**Author:** Giuliana Barbieri (ML Engineer & Student)

**Abstract:**

With the beginning of the Vera C. Rubin Observatory’s activity, we will need to classify millions of astronomical alerts every night. This is a big challenge for machine learning, especially because of the "class imbalance" problem—where some astronomical events are much more common than others.

In this talk, I present the development of **DeepRubin-Explorer**, a pipeline designed to classify four types of astronomical transients: Quasars, Cepheids, and Type Ia and II Supernovae. I will show how I started with a baseline model using a Temporal Convolutional Network (TCN) that achieved 67% accuracy but failed to identify rare supernova types.

The core of my presentation is the iterative process I followed as a student to improve these results. By tripling the size of the training dataset and implementing a "Weighted CrossEntropyLoss" to help the model learn from minority classes, I improved the peak validation accuracy to **92.16%**. I will also discuss how I used automated tools like MLflow to track these scientific experiments and why documenting the "story" of the progress is essential in time-domain astronomy. This project demonstrates that choosing the right cost function is key to building robust classification models for the LSST era.
