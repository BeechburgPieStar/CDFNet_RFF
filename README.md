# Confidence-Aware Dynamic Fusion Network for Cross-Domain Radio Frequency Fingerprinting

<img width="906" height="234" alt="image" src="https://github.com/user-attachments/assets/0f4f8c7f-0a6e-47ce-9600-d94d9d3ffebc" />

## Overview
This repository implements **CDFNet**, a Confidence-Aware Dynamic Fusion Network designed for robust radio frequency fingerprinting under cross-domain conditions.  
The model explicitly addresses performance degradation caused by receiver and channel variations by jointly exploiting **raw** and **channel-equalized** IQ signals.

CDFNet adopts a dual-branch architecture with lightweight cross-branch interaction and confidence-aware dynamic fusion, enabling adaptive integration of complementary information during inference.

---

## Evaluation Protocol

### Receiver Groups (R1–R4)
The WiSig (ManySig) dataset partitions receivers into four receiver groups, denoted as **R1, R2, R3, and R4**.

A four-fold cross-validation protocol is adopted:
- In each fold, three receiver groups are used for training
- The remaining receiver group is used for testing
- R1–R4 correspond to the four possible choices of the test receiver group

This protocol evaluates **cross-receiver generalization performance**.

---

### Cross-Channel Experimental Settings

To evaluate robustness under channel variations, three experimental settings are considered.

**Exp1**  
The model is trained using data collected on one day and tested on data from the remaining days.

**Exp2**  
The model is trained using data collected on two days and tested on data from the remaining days.

**Exp3**  
The model is trained using data collected on three days and tested on data from the remaining day.

From Exp1 to Exp3, the training domain becomes increasingly diverse, leading to reduced domain shift and improved identification performance.

---

### Avg. Metric
For each experimental setting, identification accuracy is reported for R1–R4.  
**Avg. denotes the arithmetic mean of the four receiver-group results**, and serves as the primary metric for evaluating overall cross-domain robustness.

---

## Baseline Methods

The following representative methods are included for comparison.

**TFMix**  
TFMix: A Robust Time Frequency Mixing Approach for Domain Generalization in Specific Emitter Identification  
IEEE Transactions on Cognitive Communications and Networking

**SigMix**  
SigMix: Robust Specific Emitter Identification Method Enhanced by Cross Time and Cross Receiver Mixing Augmentation  
IEEE Internet of Things Journal

**MTL**  
Channel Robust RF Fingerprint Identification Using Multi Task Learning and Receiver Collaboration  
IEEE Signal Processing Letters

**FS**  
Few Shot Cross Receiver Radio Frequency Fingerprinting Identification Based on Feature Separation  
IET Communications

**RIEI**  
Domain Generalization for Cross Receiver Radio Frequency Fingerprint Identification  
IEEE Internet of Things Journal

---

## PatchNet Baselines

**PatchNet (raw)**  
A single-branch PatchNet model using only raw IQ signals.

**PatchNet (eq)**  
A single-branch PatchNet model using only channel-equalized IQ signals.

These baselines are used to verify the individual contribution and limitations of raw and equalized signal representations.

---

## Proposed Method

**CDFNet (raw + eq)**  
The proposed Confidence-Aware Dynamic Fusion Network jointly processes raw and equalized signals via a dual-branch backbone.  
A confidence-aware dynamic fusion mechanism adaptively weights branch predictions based on energy-based confidence estimation, leading to superior robustness under cross-receiver and cross-channel conditions.

---

## How to run 

### PatchNet (Basic Model)

### CDFNet (Proposed)
Coming Soon!
