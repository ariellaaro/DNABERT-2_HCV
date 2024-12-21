# DNABERT-2 Application for HCV Genotype and Subtype Prediction

This repository is a fork of the original DNABERT-2 project. It contains modified scripts and results related to HCV genotype and subtype prediction.

Below, we provide the setup and instructions for HCV fine-tuning on a Linux OS, so you can easily replicate our experiment. For pre-training and additional information, check the [official implementation](https://github.com/MAGICS-LAB/DNABERT_2).

---

## Contents

- [1. Introduction](#1-introduction)
- [2. Data Format](#2-data-format)
- [3. Environment Setup](#3-environment-setup)
- [4. Model Setup](#4-model-setup)
- [5. Loading your fine-tuned model](#5-loading-your-fine-tuned-model)
- [6. Making Predictions](#6-making-predictions)

## 1. Introduction

DNABERT-2 is a foundation model trained on large-scale multi-species genome that achieves the state-of-the-art performance on $28$ tasks of the [GUE](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2) benchmark. It replaces k-mer tokenization with BPE, positional embedding with Attention with Linear Bias (ALiBi), and incorporate other techniques to improve the efficiency and effectiveness of DNABERT.

In this project, we fine-tuned the original DNABERT-2 using data from the [Los Alamos Sequence Database](https://hcv.lanl.gov/content/index), which was formatted using the scripts provided in the `formatting` folder.

In local evaluations, the model predicted the correct labels with over 98% accuracy and precision for the six main HCV genotypes and the most prevalent subtypes. Detailed test performance results are available in the `results` folder.

## 2. Data Format

Generate 3 csv files from your HCV dataset: `train.csv`, `dev.csv`, and `test.csv`. In the training process, the model is trained on train.csv and is evaluated on the dev.csv file. After the training if finished, the checkpoint with the smallest loss on the dev.csv file is loaded and be evaluated on test.csv.

Each file should be in the same format, with the first row as document head named `sequence, label`. Each following row should contain a DNA sequence and a numerical label concatenated by a `,` (ACGTCAGTCAGCGTACGT, 1). See [sample_data](https://github.com/MAGICS-LAB/DNABERT_2/tree/main/sample_data) for example datasets.

## 3. Environment Setup

### Anaconda

After installing [Anaconda](https://www.anaconda.com/download), run the following in the Anaconda prompt:

```shell
conda create -n dna python=3.8
conda activate dna

conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install transformers[torch]==4.29.2

conda install anaconda::scikit-learn

pip install einops
pip install peft
pip uninstall triton
```

Next, clone the official repository in the Linux shell:

```shell
cd ~
git clone https://github.com/MAGICS-LAB/DNABERT_2.git
```

## 4. Model Setup

Follow these steps before fine-tuning the model for HCV classification:

- Replace the original `train.py` and `run_dnabert2.sh` files with the versions in this repository (`DNABERT-2HCV/finetune/train.py` and `DNABERT-2HCV/finetune/scripts/run_dnabert2.sh`).
- Create a folder named `hcv_bert` in your home directory. Ensure the folder is empty before every run, as it will store the model and tokenizer.
- Create a folder named `sequences` in your home directory, and another named `hcv` inside of it. Place your input sequences inside the `hcv` folder.

In the Linux shell:

```shell
conda activate dna
python3
```

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

###

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768

# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print(embedding_max.shape) # expect to be 768
```

- `ctrl+d` to exit python

```shell
cd ~/DNABERT_2/finetune

sh scripts/run_dnabert2.sh $DATA_PATH

```

## 5. Loading your fine-tuned model

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

hcv_bert = os.path.expanduser("~/hcv_bert")

tokenizer = AutoTokenizer.from_pretrained(hcv_bert, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(hcv_bert, trust_remote_code=True)
```

## 6. Making Predictions

### For single sequence classification:

- Replace "ACGTACGTACGT" with your own sequence.

```python
seq = "ACGTACGTACGT"

inputs = tokenizer(seq, return_tensors="pt")

with torch.no_grad():
	logits = model(**inputs).logits
	predicted_class_id = torch.argmax(logits, dim=-1).item()
	label = model.config.id2label[predicted_class_id]

print(f"Label: {label}")

```

### For multiple sequence classification:

- Place your sequences.csv file in the home directory, with one sequence per line (no header or labels).

```python
pip install pandas
import pandas as pd

csv_path = os.path.expanduser("~/sequences.csv")

df = pd.read_csv(csv_path, header=None)
sequences = df[0].tolist()

predicted_labels = []

for seq in sequences:
    inputs = tokenizer(seq, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[predicted_class_id]
        predicted_labels.append(label)

df['Predicted_Label'] = predicted_labels
output_csv_path = os.path.expanduser("~/predicted_labels.csv")
df.to_csv(output_csv_path, index=False, header=["Sequence", "Predicted_Label"])

print(f"Predicted labels saved in: {output_csv_path}")

```
