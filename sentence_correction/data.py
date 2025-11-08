import os
import nltk
nltk.download('brown')
from nltk.corpus import brown

print(brown.words())

def tokenize(text):
	#nltk.data.path.append(os.path.dirname(os.path.dirname(__file__)))
	nltk.data.path.append(os.path.dirname(__file__))
	return nltk.tokenize.word_tokenize(text)