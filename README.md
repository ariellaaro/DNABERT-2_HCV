# DNABERT-2 Application for HCV Genotype and Subtype Prediction

This repository is a fork of the original DNABERT-2 project. It contains modified scripts and results related to HCV genotype and subtype prediction.

Below is the recommended setup for HCV fine-tuning using a Linux OS. Please check and edit your own file/folder paths before running the following codes.

For pre-training information, as well as a guide on how to fine-tune the model on your own datasets, check the [official implementation](https://github.com/MAGICS-LAB/DNABERT_2).

---

## Contents

- [1. Introduction](#1-introduction)
- [2. Environment Setup](#2-environment-setup)
- [3. Model Setup](#3-model-setup)
- [4. Saving and loading fine-tuned model](#4-saving-and-loading-fine-tuned-model)
- [5. Making Predictions](#5-making-predictions)

## 1. Introduction

DNABERT-2 is a foundation model trained on large-scale multi-species genome that achieves the state-of-the-art performance on $28$ tasks of the [GUE](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2) benchmark. It replaces k-mer tokenization with BPE, positional embedding with Attention with Linear Bias (ALiBi), and incorporate other techniques to improve the efficiency and effectiveness of DNABERT.

## 2. Environment Setup

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

## 3. Model Setup

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

## 4. Saving and loading fine-tuned model


```python
'''
before fine-tuning:

- create the folder "hcv_bert" in the home directory
- add the following code in train.py, before the lines:

if __name__ == "__main__":
    train()
'''

hcv_bert = os.path.expanduser("~/hcv_bert")

model.save_pretrained(hcv_bert)
print(f"Saved in {hcv_bert}")

tokenizer.save_pretrained(hcv_bert)
print(f"Saved in {hcv_bert}")
```

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

hcv_bert = os.path.expanduser("~/hcv_bert")

tokenizer = AutoTokenizer.from_pretrained(hcv_bert, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(hcv_bert, trust_remote_code=True)
```

## 5. Making Predictions

- For single sequence classification:

```python
seq = "ACGTACGTACGT"

inputs = tokenizer(seq, return_tensors="pt")

with torch.no_grad():
	logits = model(**inputs).logits
	predicted_class_id = torch.argmax(logits, dim=-1).item()
	label = model.config.id2label[predicted_class_id]

print(f"Label: {label}")

```

- For multiple sequence classification (from a csv file):

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
