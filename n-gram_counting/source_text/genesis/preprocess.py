# TODO: (For more general text forms, split by sentence. Rn it's already one line per sentence so dw)
#from nltk.tokenize import sent_tokenize

input_file = "genesis.txt"
output_file = "preprocessed.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.split("\t")
        if len(parts) > 1:
            text = parts[1].strip()
            outfile.write(text + "\n")
