# Radiology Report Analyzer

## Overview

This repository contains code and documentation for a class project that builds and compares Convolutional Neural Network (CNN) models to automatically classify chest‑X‑ray images.  Radiology departments often experience large report backlogs; a 2024 survey of radiology practices found that about **68 %** of practices had unreported exams and the median turnaround from exam completion to radiologist interpretation was over **520 minutes (~8.7 hours)**【413055050020672†screenshot】.  Automating the initial interpretation of routine chest studies can therefore accelerate reporting and allow radiologists to concentrate on complex cases.

The project evaluates two CNN architectures—**EfficientNet‑B0** and **ResNet‑18**—using transfer learning.  The models were trained to identify three common pathologies (Atelectasis, Effusion and Infiltration) on subsets of the NIH Chest X‑ray database.  A lightweight EfficientNet‑B0 model trained on ~17 k images achieved a final test accuracy of about **69.9 %** and a macro F1‑score of **0.70**【48998811865785†screenshot】.  A ResNet‑18 baseline trained on ~27 k images was early‑stopped after six epochs and achieved ~64 % accuracy with a macro F1‑score of **0.64**【483618361653168†screenshot】.  These results show that EfficientNet‑B0 delivers better overall performance even when trained on a smaller dataset【439580607207394†screenshot】.

## Business Objective

Radiology services are under pressure to manage increasing image volumes.  The backlog of unreported studies delays diagnoses and contributes to clinician burnout.  By building a reliable classifier that labels incoming chest‑X‑rays and pre‑populates draft reports, this project aims to reduce reporting times and free radiologists to focus on ambiguous cases【413055050020672†screenshot】.

## Data

We used publicly available data from the **NIH Chest X‑ray** dataset.  Images with a single label belonging to one of three classes—Atelectasis, Effusion and Infiltration—were extracted.  The data were split into training, validation and test sets as summarised below【64686200411970†screenshot】:

| Experiment                    | Total images | Train | Validation | Test |
|-------------------------------|-------------:|------:|----------:|-----:|
| EfficientNet‑B0 (primary)     | 17,717       | 12,401 | 2,657      | 2,659 |
| ResNet‑18 (baseline)          | 27,409       | 19,185 | 4,111      | 4,112 |

Class imbalance was addressed by computing inverse‑frequency class weights and using a weighted cross‑entropy loss during training【751429311156171†screenshot】.

## Methods

Both models leverage transfer learning from pretrained ImageNet weights.  The key components are:

* **Model architectures** – EfficientNet‑B0 scales depth, width and resolution using compound scaling.  ResNet‑18 uses residual blocks with skip connections.  Only the final classifier layer was replaced to output three classes【751429311156171†screenshot】.
* **Training** – Images are resized to a consistent resolution and randomly augmented.  We used the AdamW optimizer with an initial learning rate of `1e‑4` and trained up to 20 epochs with early stopping (patience = 3) based on validation loss.  Weighted cross‑entropy loss mitigated class imbalance【751429311156171†screenshot】.
* **Evaluation** – Overall accuracy and macro F1‑score were used to quantify performance.  Confusion matrices and per‑class precision/recall/f1 metrics were reported.  Grad‑CAM heatmaps provided qualitative insights into which lung regions contributed to predictions【483618361653168†screenshot】.

## Results

**EfficientNet‑B0:** On a held‑out test set of 2,659 images, the EfficientNet‑B0 model achieved **69.91 %** accuracy with the following per‑class F1‑scores【48998811865785†screenshot】:

| Class        | Precision | Recall | F1‑score |
|--------------|---------:|-------:|---------:|
| Atelectasis  | 0.58     | 0.56   | 0.57     |
| Effusion     | 0.66     | 0.63   | 0.64     |
| Infiltration | 0.76     | 0.80   | 0.78     |
| **Macro avg** | –         | –       | **0.70** |

**ResNet‑18:** The ResNet‑18 experiment was halted after six epochs when the validation loss stopped improving.  On its test set of 4,112 images it achieved **~64 %** accuracy and the following per‑class F1‑scores【483618361653168†screenshot】:

| Class        | Precision | Recall | F1‑score |
|--------------|---------:|-------:|---------:|
| Atelectasis  | 0.52     | 0.45   | 0.48     |
| Effusion     | 0.59     | 0.58   | 0.59     |
| Infiltration | 0.71     | 0.76   | 0.73     |
| **Macro avg** | –         | –       | **0.64** |

The EfficientNet‑B0 model outperformed ResNet‑18 across all classes despite being trained on fewer images, leading us to select it for deployment【439580607207394†screenshot】.

## Repository Structure

```
radiology‑report‑analyzer/
├── Efficient_Net_CNN_Model_X_Ray_Image_Classification_Group_4.ipynb  # Jupyter notebook training EfficientNet‑B0
├── ResNet_Model_X_Ray_Image_Classification_Group_4.ipynb             # Jupyter notebook training ResNet‑18
├── Radiology XRay Report.pdf                                        # Project report detailing methodology and results
├── README.md                                                        # This file
├── requirements.txt                                                 # Python dependencies
```

### requirements.txt

The `requirements.txt` file lists the core Python packages used in the notebooks.  A typical environment includes:

```
torch==1.12.0
torchvision==0.13.0
numpy
pandas
matplotlib
seaborn
scikit‑learn
tqdm
jupyter
``` 

You can adjust the versions based on your system; the versions above were tested during development.

## Getting Started

1. **Clone the repository** (once you have published it to GitHub).
   ```bash
   git clone https://github.com/your‑username/radiology‑report‑analyzer.git
   cd radiology‑report‑analyzer
   ```

2. **Create a Python environment** and install the dependencies.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download the dataset.** The NIH Chest X‑ray dataset is large and cannot be included in this repository.  You can download it from the NIH open data portal (https://nihcc.app.box.com/v/ChestXray-NIHCC).  Filter the images to the three target labels as described in the notebooks【64686200411970†screenshot】.

4. **Run the notebooks.** Open Jupyter and run either notebook to reproduce the experiments.  You may need to update the dataset paths in the code.  The notebooks will train the models, plot learning curves and output classification reports.

## License

This project is released under the MIT License.  See the `LICENSE` file for details (you may create one before publishing).

## Authors

This project was completed by Group 4 (five members) as part of a data science course.  We thank the NIH for providing the chest‑X‑ray dataset and acknowledge open‑source contributors whose libraries made this work possible.
