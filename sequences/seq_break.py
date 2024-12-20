# sequences with 1000 nucleotides max, some with less

import pandas as pd

def split_sequence(sequence, label, max_length=1000):
    pedacos = [sequence[i:i+max_length] for i in range(0, len(sequence), max_length)]
    return [(frag, label) for frag in pedacos]

input_path = 'C:/Users/your_user/Downloads/HCV.csv'
output_path = 'C:/Users/your_user/Downloads/HCV_break.csv'

df = pd.read_csv(input_path, header=None, names=['sequence', 'label'], skiprows=1)

seq_menores = []

for index, row in df.iterrows():
    sequence = row['sequence']
    label = row['label']
    pedacos = split_sequence(sequence, label)
    seq_menores.extend(pedacos)

df_fragments = pd.DataFrame(seq_menores, columns=['sequence', 'label'])

header = pd.read_csv(input_path, nrows=1, header=None)
header.to_csv(output_path, index=False, header=False, mode='w')
df_fragments.to_csv(output_path, index=False, header=False, mode='a')
