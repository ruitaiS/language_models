import data
import random
import math

unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
xft, tfx = data.get_lookups()

#l1, l2, l3 = (0.14051555, 0.48004991, 0.37943454)
l1, l2, l3 = (0.11508279, 0.50076896, 0.38414825)
fallback = -100

def output(max_tokens=200):
	prob_distribution = {}
	sentence = [xft['<s>']]
	while len(sentence) < max_tokens:
		for index in tfx.keys(): # Iterate over all possible next words
			if len(sentence) == 1:
				bigram_lp = bigram.get((xft['<s>'], index), fallback)
				prob = l2 * (10**bigram_lp)
			else:
				unigram_lp = unigram.get(index, fallback)
				bigram_lp = bigram.get((sentence[-1], index), fallback)
				trigram_lp = trigram.get((sentence[-2], sentence[-1], index), fallback)
				print(f"Type: {type(index)}, value: {index}")
				if int(index) <= 10: # Ignore unigram probs for punctuation
					prob = l2 * (10**bigram_lp) + l3 * (10**trigram_lp)
				else:
					prob = l1 * (10**unigram_lp) + l2 * (10**bigram_lp) + l3 * (10**trigram_lp)
			prob_distribution[index] = prob ## TODO: Verify super janky softmax
		#print(sum(prob_distribution.values()))

		# TODO: This sampling method is psychotic and mathematically unsound
		leftover = random.uniform(0, sum(prob_distribution.values()))
		print(sum(prob_distribution.values()))
		for index, prob in prob_distribution.items():
			leftover -= prob
			if leftover <= 0 :
				sentence.append(index)
				#print(sentence)
				break
		
		if sentence[-1] == xft['</s>']:
			break
	return sentence
			
index_list = output()
readable_list = [tfx[index] for index in index_list]
print(" ".join(readable_list))