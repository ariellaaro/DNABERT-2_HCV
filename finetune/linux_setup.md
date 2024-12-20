> Please check and edit file/folder paths before running the following codes.

### Anaconda Setup

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

```shell
cd ~
git clone https://github.com/MAGICS-LAB/DNABERT_2.git
```

---

### Running DNABERT_2

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

from transformers.models.bert.configuration_bert import BertConfig

config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

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
cd ~/dna/DNABERT_2/finetune

sh scripts/hcv.sh $DATA_PATH

```

---

### Saving and loading the finetuned model

```python
'''
add the following code in train.py, before the following:

if __name__ == "__main__":
    train()
'''
# caution!! create the folder "hcv_bert" at the home directory

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

---

### Making predictions

#### Single sequence

```python
seq = "ACGTACGTACGT"

inputs = tokenizer(seq, return_tensors="pt")

with torch.no_grad():
	logits = model(**inputs).logits
	predicted_class_id = torch.argmax(logits, dim=-1).item()
	label = model.config.id2label[predicted_class_id]

print(f"Label: {label}")

```

#### Multiple sequences (from a csv file)

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
