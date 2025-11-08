import os
import nltk

def tokenize(text, method = 'nltk'):
	if (method == 'nltk'):
		nltk.data.path.append(os.path.dirname(os.path.dirname(__file__)))
		return nltk.tokenize.word_tokenize(text)
	elif (method == 'bpe'):
		print('bpe todo')
		raise Exception('bpe')