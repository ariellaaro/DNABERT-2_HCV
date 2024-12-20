import csv

def fasta(file_path):
    sequences = []
    labels = []

    with open(file_path, 'r') as file:
        current_sequence = ""

        for line in file:
            line = line.strip()

            if line.startswith(">"):

                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ""
                label = line[1]
                labels.append(label)

            else:
                current_sequence += ''.join([char for char in line if char in "ATCG"])

        if current_sequence:
            sequences.append(current_sequence)

    return labels, sequences

def save_csv(labels, sequences, output_file):

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sequence', 'label'])
        for seq, lbl in zip(sequences, labels):
            writer.writerow([seq, lbl])

fasta_path = 'C:/Users/your_user/Downloads/HCV.fasta'
csv_path = 'C:/Users/your_user/Downloads/HCV.csv'

labels, sequences = fasta(fasta_path)

save_csv(labels, sequences, csv_path)
