import os
import nltk
nltk.data.path.append(os.path.dirname(__file__))
from nltk.tokenize import word_tokenize

filename = "text/a3_test_set.txt" # "text/a2_dev_set.txt"
with open(filename, "r") as test_file:
	for line in test_file:
		parts = line.split("\t")
		if len(parts) > 1:
			text = parts[1].strip()
			tokens = word_tokenize(text)
			print(tokens)