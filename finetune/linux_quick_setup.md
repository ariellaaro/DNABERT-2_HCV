> For when you already have the model ready to finetune. Change file/folder paths as needed.

```shell
conda activate dna
python3
```

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
```

- `ctrl+d` to exit python

```shell
cd ~/DNABERT_2

sh scripts/hcv.sh
```

---

## After finetuning

#### Loading model

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

hcv_bert = os.path.expanduser("~/saved")
tokenizer = AutoTokenizer.from_pretrained(hcv_bert, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(hcv_bert, trust_remote_code=True)

```

#### Single sequence prediction

```python
seq = "ACGTACGTACGT"

inputs = tokenizer(seq, return_tensors="pt")

with torch.no_grad():
	logits = model(**inputs).logits
	predicted_class_id = torch.argmax(logits, dim=-1).item()
	label = model.config.id2label[predicted_class_id]

print(f"Label: {label}")

```

#### Multiple sequence prediction

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

print(f"Predicted labels saved in {output_csv_path}")
```
