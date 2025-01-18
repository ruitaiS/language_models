# TODO:
#from nltk.tokenize import sent_tokenize

input_file = "genesis.txt"
output_file = "preprocessed.txt"

# Open the input file for reading and output file for writing
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Split the line on the tab character
        parts = line.split("\t")
        if len(parts) > 1:
            # Extract the text after the tab and strip any leading/trailing whitespace
            text = parts[1].strip()
            # Write the text to the output file
            outfile.write(text + "\n")
