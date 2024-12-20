# sequences with 999 nucleotides only; the ends are discarded

import pandas as pd

def split_sequence(sequence, label, max_length=999):
    pedacos = [sequence[i:i+max_length] for i in range(0, len(sequence), max_length) if len(sequence[i:i+max_length]) == max_length]
    return [(frag, label) for frag in pedacos]

input_path = 'C:/Users/your_user/Downloads/HCV.csv'
output_path = 'C:/Users/your_user/Downloads/HCV_break.csv'

df = pd.read_csv(input_path, header=None, names=['sequence', 'label'], skiprows=1)

seq_999 = []

for index, row in df.iterrows():
    sequence = row['sequence']
    label = row['label']
    pedacos = split_sequence(sequence, label)
    seq_999.extend(pedacos)

df_fragments = pd.DataFrame(seq_999, columns=['sequence', 'label'])

df_fragments.to_csv(output_path, index=False, header=False)
