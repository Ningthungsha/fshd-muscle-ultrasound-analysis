# FSHD Muscle Ultrasound Analysis using Hybrid Deep Learning

Automated Heckmatt grading and muscle quality assessment in **Facioscapulohumeral Muscular Dystrophy (FSHD)** using musculoskeletal ultrasound images.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-189fdd)](https://xgboost.ai)

---

## 📌 Project Overview

**Facioscapulohumeral Muscular Dystrophy (FSHD)** is a genetic muscle disease where healthy muscle tissue gradually degenerates and gets replaced by fat and fibrous collagen. 

Doctors assess disease severity using musculoskeletal B-mode ultrasound and the **Heckmatt grading scale (Grades 1 to 4)**. However, manual grading is subjective and prone to variation between doctors, ultrasound gain differences, and skin/gel artifacts.

This project implements a hybrid machine learning pipeline that:
1. **Segments the muscle region** using an **Attention U-Net** to remove skin, transducer gel, and subcutaneous fat.
2. **Extracts 89 standardized radiomic features** (first-order intensity, GLCM, GLRLM, GLSZM texture, and deep bone shadow attenuation markers).
3. **Learns spatial features** via a modified grayscale **ResNet-18** (128-dimensional latent vector).
4. **Fuses features dynamically** using a **Gated Attention Network** (Sigmoid gate) that balances deep features against tabular radiomics.
5. **Predicts the Heckmatt grade (1-4)** and disease severity using a weighted consensus committee (Hybrid Network + XGBoost + Random Forest).
6. **Provides Explainability (XAI)** using Layer-4 Grad-CAM heatmaps and TreeSHAP feature attributions.

---

##  Heckmatt Clinical Grading Scale

| Grade | Disease Status | Muscle Ultrasound Appearance | Deep Bone Reflection |
| :---: | :--- | :--- | :--- |
| **Grade 1** | **Normal** | Dark (hypoechoic) healthy muscle with organized fascial lines | **Sharp & Bright:** Sound wave reflects strongly from the bone |
| **Grade 2** | **Mild** | Slight increase in brightness, scattered white speckles | **Distinct:** Bone reflection remains clear |
| **Grade 3** | **Moderate** | Significant diffuse ground-glass brightness across the muscle | **Reduced:** Bone reflection is noticeably faded / attenuated |
| **Grade 4** | **Severe / End-Stage** | Dense white fibrous replacement throughout the muscle | **Extinguished:** Complete loss/shadowing of bone reflection |

---

##  Web Application (Streamlit Dashboard)

The web dashboard is built with a Neo-Brutalist interface divided into 5 modules:

### 1. Clinical Scanner & Predictor
Upload any B-mode ultrasound image, run real-time Attention U-Net epimysium segmentation, extract radiomics, and view predicted Heckmatt grade, class confidence, and a continuous severity score (1.0 to 4.0).

![Clinical Scanner](docs/screenshots/01_clinical_scanner.png)

---

### 2. Model Benchmark & Ablations
Comparison of models evaluated under 5-Fold Stratified GroupKFold cross-validation (grouped by patient ID) with quadratic error confusion matrices.

![Benchmark Ablation](docs/screenshots/02_benchmark_ablation.png)

---

### 3. Explainable AI (Grad-CAM & SHAP)
Layer-4 Grad-CAM saliency heatmaps verify that predictions focus on deep muscle tissue rather than superficial gel, alongside TreeSHAP feature importance plots.

![Explainable AI](docs/screenshots/03_explainable_ai.png)

---

### 4. Machine Gain Drift & Stress-Testing
Simulates real-world ultrasound gain fluctuations (±20%) and speckle noise to test prediction stability.

![Domain Generalization](docs/screenshots/04_domain_generalization.png)

---

### 5. Patient Analytics & Report Export
Aggregates scan results across multiple muscles and generates downloadable diagnostic summaries.

![Patient Report](docs/screenshots/05_patient_report.png)

---

##  How It Works (Pipeline)

![End-to-End System Architecture Pipeline](docs/system_architecture.png)

The automated diagnostic workflow processes raw ultrasound B-mode scans across six modular stages:

1. **Input & Preprocessing:** 512×512 raw transverse sonograms undergo min-max intensity normalization and acoustic gain drift correction.
2. **Epimysium Segmentation (Attention U-Net):** Additive Attention Gates ($\alpha = \sigma(W_g g + W_x x + b)$) suppress skin, bone reflection, and acoustic gel artifacts, isolating the muscle epimysium ROI (91.4% Dice score).
3. **Dual-Stream Feature Extraction:**
   - **Stream 3A (PyRadiomics):** 89 quantitative acoustic descriptors (18 first-order moments, 24 GLCM, 16 GLRLM, 16 GLSZM, 5 morphometric CSA, and 10 bone shadow extinction markers).
   - **Stream 3B (Deep Representation):** 1-channel modified ResNet-18 extracting a 128-dimensional spatial latent vector ($h_{deep} \in \mathbb{R}^{128}$).
4. **Cross-Modal Gated Attention Fusion:** Projects deep and radiomic features to a 64-d common space with a dynamic gating head ($\alpha \in [0, 1]$) that adaptively balances spatial deep learning with quantitative radiomics.
5. **Consensus Committee Ensemble:** Soft probability voting integrating the Gated Hybrid (40%), Pure-Vision XGBoost (35%), and Pure-Vision Random Forest (25%) achieving a Quadratic Weighted Kappa of 0.828 ± 0.02.
6. **Clinical Decision & Explainable AI:** Outputs 4-class ordinal Heckmatt grades (1–4), continuous severity scores (1.00–4.00), Layer-4 Grad-CAM activation heatmaps, and TreeSHAP attribution waterfalls.

---

##  Benchmark Results

Evaluated on 361 clinical scans across 110 subjects using 5-Fold Stratified GroupKFold (patient-level split to prevent data leakage):

| Model Architecture | Accuracy | Balanced Accuracy | Macro F1 | Quadratic Weighted Kappa ($\kappa_w$) | Ordinal MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 (Pixels Only)** | 52.3% | 50.7% | 0.468 | 0.548 | 0.62 |
| **XGBoost (Radiomics)** | 66.2% | 65.4% | 0.646 | 0.807 | 0.36 |
| **Random Forest (Radiomics)** | 68.4% | 66.8% | 0.664 | 0.813 | 0.34 |
| **Early Fusion (Concatenation)** | 67.3% | 67.2% | 0.657 | 0.818 | 0.36 |
| **Late Fusion Ensemble** | 66.8% | 65.8% | 0.651 | 0.810 | 0.35 |
| **Proposed Gated Hybrid** | **65.7%** | **64.5%** | **0.637** | **0.828** | **0.36** |
| **Ensemble Committee (External Scans)** | **75.0%** | **74.2%** | **0.720** | **0.857** | **0.25** |

*Note: Quadratic Weighted Kappa ($\kappa_w$) penalizes large misclassifications (e.g. Grade 1 vs Grade 4) more heavily than adjacent grades.*

---

##  Repository Structure

```
.
├── app.py                      # Main Streamlit web dashboard
├── train_pure_vision_engine.py # Training pipeline with acoustic augmentation
├── run_benchmark.py            # 5-fold cross-validation runner
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── docs/
│   └── screenshots/            # Dashboard UI screenshots
│
└── src/
    ├── segmentation/
    │   ├── models.py           # Attention U-Net & AttentionGate
    │   ├── losses.py           # DiceBCELoss & FocalDiceLoss
    │   └── metrics.py          # Dice, IoU, Precision, Recall
    ├── radiomics/
    │   └── extractor.py        # 89-feature radiomic extractor
    ├── deep_features/
    │   ├── backbones.py        # Grayscale ResNet-18 feature extractor
    │   └── cam.py              # Layer-4 Grad-CAM heatmap generator
    ├── fusion/
    │   ├── models.py           # GatedAttentionFusionModel & baselines
    │   └── ordinal_loss.py     # OrdinalHeckmattLoss
    ├── evaluation/
    │   └── metrics.py          # Classification metrics (QWK, BAcc, MAE)
    └── explainability/
        └── shap_explainer.py   # TreeSHAP feature attribution module
```

---

##  Getting Started

### 1. Clone & Setup Environment

```bash
git clone https://github.com/rajrathod1233/fshd-muscle-ultrasound-analysis.git
cd fshd-muscle-ultrasound-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web App

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Run Benchmark / Training

```bash
# Run 5-fold cross validation benchmark
python run_benchmark.py

# Re-train models with data augmentation
python train_pure_vision_engine.py
```

---

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Machine Learning:** Scikit-Learn, XGBoost
- **Medical Feature Extraction:** PyRadiomics, Scikit-Image, OpenCV
- **Web App:** Streamlit, Plotly
- **Data & Processing:** NumPy, Pandas, SciPy
---

## Team Members

This project was developed collaboratively by the following team members:

1. [Ningthungsha](https://github.com/Ningthungsha)
2. [Raj Rathod](https://github.com/rajrathod1233)
3. [Borngamsha](https://github.com/borngamsha)
4. [Wangli](https://github.com/Wangli762)
5. [Pritam Das](https://github.com/pritamdas7)
6. [Roy Khartu](https://github.com/RoyKhartu-sys)

