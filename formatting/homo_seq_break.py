import csv
import math

def process_sequences(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            sequence, label = row[0], row[1]
            seq_len = len(sequence)
            chunk_size = max(1, math.ceil(seq_len / 10))

            for i in range(0, seq_len, chunk_size):
                chunk = sequence[i:i + chunk_size]
                writer.writerow([chunk, label])

input_file = 'C:/Users/your_user/Downloads/HCV.csv'
output_file = 'C:/Users/your_user/Downloads/HCV_break.csv'
process_sequences(input_file, output_file)
