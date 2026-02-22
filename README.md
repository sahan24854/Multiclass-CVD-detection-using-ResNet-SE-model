# Cardiovascular Disease Detection from 12-Lead ECG
## Project Overview
This project evaluates the efficacy of 1D Convolutional Neural Networks (CNN) using Residual Architectures for the automated interpretation of 12-lead Electrocardiograms (ECG). The goal was to investigate how deep residual learning can extract high level morphological features from raw waveforms to predict major cardiovascular disease (CVD) superclasses.

## Dataset: PTB-XL, a large publicly available electrocardiography dataset
The PTB-XL dataset is currently the largest freely accessible clinical 12-lead ECG waveform dataset. It comprises 21,837 clinical records of 10 seconds each from 18,885 patients.

* Multi Label Complexity: Records are annotated with up to two cardiologists using 71 unique statements.
* Hierarchical Labels: These statements are aggregated into 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP) for coarse grained classification.
* Diverse Quality: The dataset reflects real-world clinical conditions, featuring varied signal quality, including baseline drift and static noise.

## Signal and Data Preprocessing Pipeline
To transform raw medical waveforms into model ready tensors, a robust preprocessing pipeline was implemented. This stage is critical for ensuring the ResNet architecture focuses on clinical morphologies rather than recording artifacts.

1. Signal Acquisition & Resampling
   
Raw data is converted from the binary WFDB format. While the dataset is available at 500 Hz, this project utilizes the 100 Hz downsampled version to maintain a standardized input shape of (1000, 12) for 10 second recordings.

2. Artifact Removal (Filtering)
   
ECG signals are inherently noisy. A 3rd order Butterworth Bandpass Filter (0.5Hz - 45Hz) is applied to:
* Eliminate Baseline Wander: Removing low-frequency drift caused by patient respiration or movement.
* Suppress Powerline Interference: Reducing high-frequency 50/60Hz noise (static noise).

3. Lead wise Normalization
   
To ensure the model is invariant to differing signal amplitudes across patients, Z score normalization is performed independently for each of the 12 leads. This standardizes the signal to have a zero mean and unit variance, which accelerates convergence in deep residual layers.

5. Diagnostic Aggregation & Labeling
   
Using the scp_statements.csv metadata, the fine-grained diagnostic labels are mapped to the 5 primary superclasses. This creates a Multi-Hot encoded target, acknowledging that a single patient may present with multiple co-occurring cardiovascular conditions.

## Model Architecture

### 1. Modular Residual Blocks with SE Attention

The core of the model is a custom Residual Block integrated with a Squeeze and Excitation (SE) attention mechanism.
* Residual Connections: These allow for training deeper networks by preventing vanishing gradients, enabling the model to learn complex morphological patterns.
* SE Blocks: I implemented lead-wise attention to dynamically weight the importance of each of the 12 ECG leads, helping the model focus on the most diagnostic channels for specific pathologies

### 2. Advanced Feature Extraction - Integrating CBAM
 
To systematically reduce avoidable bias and improve the sensitivity of the model, I transitioned from simple Squeeze and Excitation (SE) to CBAM (Convolutional Block Attention Module). This was critical for capturing the multi dimensional nature of 12-lead ECG signals.

#### Dual-Attention Mechanism
Unlike the SE-block which only focuses on lead importance, the CBAM blocks I implemented perform two sequential operations:
* Channel Attention: Identifies which of the 12 leads contain the most relevant diagnostic features.
* Spatial Attention: Focuses on the temporal axis (the 1000 time steps), allowing the model to "lock onto" specific segments of the heartbeat, such as the ST-segment or the QRS complex, where pathologies typically manifest.
