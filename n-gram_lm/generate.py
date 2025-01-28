import data
import random
import math

unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
xft, tfx = data.get_lookups()

#l1, l2, l3 = (0.14051555, 0.48004991, 0.37943454)
#l1, l2, l3 = (0.11508279, 0.50076896, 0.38414825)
#l1, l2, l3 = (0.11508205, 0.50076979, 0.38414816)
l1, l2, l3 = (0.11519852, 0.50076749, 0.38403399)
fallback = -100

def output(max_tokens=200):
	sentence = [xft['<s>']]
	prob_distribution = {} # TODO: figure out why this works
	while len(sentence) < max_tokens:
		for index in tfx.keys(): # Iterate over all possible next words
			# prob_distribution = {} # TODO: when really it should be here instead
			if len(sentence) == 1:
				bigram_lp = bigram.get((xft['<s>'], index), fallback)
				prob = l2 * (10**bigram_lp)
			else:
				unigram_lp = unigram.get(index, fallback)
				bigram_lp = bigram.get((sentence[-1], index), fallback)
				trigram_lp = trigram.get((sentence[-2], sentence[-1], index), fallback)
				if int(index) <= 10 or int(index) == 9833: # Ignore unigram probs for punctuation and 's
					prob = l2 * (10**bigram_lp) + l3 * (10**trigram_lp)
				else:
					prob = l1 * (10**unigram_lp) + l2 * (10**bigram_lp) + l3 * (10**trigram_lp)
			prob_distribution[index] = prob ## TODO: Verify super janky softmax

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

sentences = []
for i in range(10):
	index_list = output()
	readable_list = [tfx[index] for index in index_list]
	sentences.append(" ".join(readable_list))
print(sentences)# for sentence in sentences