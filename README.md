---
title: Arabidopsis Promoter Prediction
emoji: 💻
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 5.13.2
app_file: app.py
pinned: false
license: mit
short_description: A sequence-based promoter predictor for Arabidopsis Thaliana
---

# Promoter Classification using a CNN Approach for *Arabidopsis thaliana*

## Overview
Promoter sequences play a crucial role in gene transcription regulation, making their accurate identification essential for various biological applications, including crop improvement and stress resistance. Traditional promoter identification methods often rely on known motifs and positional patterns, limiting their ability to detect complex or atypical promoter sequences.

This project leverages a Convolutional Neural Network (CNN) to classify DNA sequences from *Arabidopsis thaliana* as either promoters or non-promoters. The model is trained on a combination of real and synthetic sequences, aiming to enhance promoter prediction capabilities. This is hosted via GRadio on HuggingFace located here: https://huggingface.co/spaces/meekale/Arabidopsis_Promoter_Prediction or can be locally ran.

<p align="center">
    <img src="https://github.com/user-attachments/assets/402c5048-bcc1-4622-bfcd-d431c949400c" width="800">
</p>

---
## Problem Statement
Current promoter identification methods depend on known motifs and structural patterns, which limits their effectiveness in detecting novel or non-canonical promoter regions. My goal is to develop a machine-learning model capable of accurately classifying promoters without relying on predefined motifs.

### Solution Approach
I implemented a CNN-based classifier that:
- Takes in fixed-length DNA sequences (600 bp)
- Uses one-hot encoding to transform nucleotide sequences into numerical matrices
- Extracts features through convolutional layers optimized to detect motifs
- Integrates GC content as an additional feature to improve classification
- Employs dropout regularization to mitigate overfitting

---
## Model Architecture
The model processes DNA sequences through the following layers:
1. **One-hot encoding:** Converts nucleotide sequences into a 4-channel matrix.
2. **Convolutional layers:** Detects patterns and motifs within sequences.
3. **Pooling layers:** Reduces dimensionality and enhances feature extraction.
4. **Fully connected layers:** Classifies sequences using sigmoid activation.

### Hyperparameter Optimization
Using the Hyperband tuning algorithm, the following ranges were explored:
- Convolutional filters: 32 - 256
- Kernel sizes: 3 - 20
- Pooling sizes: 2 - 4
- Dense units: 64 - 512
- Dropout rates: 0.25 - 0.5
- Learning rates: 1e-6 to 1e-3 (log-sampled)

---
## Data Sources
- **Eukaryotic Promoter Database (EPD):** Real promoter sequences.
- **Synthetic sequences:** Generated using methods inspired by Oubounyt et al. (2019), designed to mimic base composition while lacking functional motifs.
- **Preprocessing:** Ensured all sequences contained only valid nucleotides (A, T, C, G) and standardized sequence lengths to 600 bp.

---
## Model Performance
The model achieved strong classification results with minimal misclassification:

|  | Actually Positive | Actually Negative |
|---|---|---|
| **Predicted Positive** | 4477 | 32 |
| **Predicted Negative** | 97 | 4274 |

With an accuracy exceeding 94%, the model demonstrates strong predictive power for promoter classification.

### Training Performance
Below is a visualization of model accuracy and loss during training:

![model_accuracy_and_loss](https://github.com/user-attachments/assets/4f414b2a-423a-4681-8a3c-e5b396c35193)
*Figure: Training Accuracy (Left) and Loss (Right)*

Despite minor spikes in validation loss at epochs 4 and 15, the model remains effective, though further fine-tuning of learning rates and dropout regularization may improve generalizability.

---
## Future Work
- **Generalization to other organisms:** Expanding training data to other species.
- **Motif detection analysis:** Evaluating which sequence features contribute most to classification.
- **Hyperparameter tuning:** Further refining model architecture to mitigate overfitting.

---
## References
1. [Eukaryotic Promoter Database](https://epd.expasy.org/epd/arabidopsis/arabidopsis_database.php?db=arabidopsis)
2. Roy & Singer (2015). *Core promoters in transcription: old problem, new insights.* Trends in Biochemical Sciences.
3. Oubounyt et al. (2019). *Prediction of promoters and enhancers using deep learning.* Frontiers in Genetics.

## Conclusion
This project demonstrates the feasibility of CNN-based promoter classification in *Arabidopsis thaliana*. With a robust dataset and deep learning approach, our model effectively identifies promoter sequences, presenting a valuable tool for genomic research. Future improvements could enhance generalizability and interpretability, making this a promising step toward more advanced promoter prediction models.


---
