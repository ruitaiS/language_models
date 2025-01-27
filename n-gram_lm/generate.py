import data
import random

unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
xft, tfx = data.get_lookups() # index from token, token from index

#l1, l2, l3 = (9.49650121e-01, 9.37157253e-18, 5.03498791e-02)

#l1 = 0.19294876040543507
#l2 = 0.8070512544957388
#l3 = 0.0

#l1, l2, l3=(1.57446217e-01, 8.42553783e-01, 1.36187719e-14)

l1, l2, l3 = (0.14434065, 0.48567127, 0.36998808)
fallback = -100

def output(max_tokens=200):
	prob_distribution = {}
	sentence = [xft['<s>']]
	while len(sentence) < max_tokens:
		for index in tfx.keys():
			unigram_lp = unigram.get(index, fallback)
			if len(sentence) == 1:
				bigram_lp = bigram.get((xft['<s>'], index), fallback)
				prob = l1 * (10**unigram_lp) + l2 * (10**bigram_lp)
			else:
				bigram_lp = bigram.get((sentence[-1], index), fallback)
				trigram_lp = trigram.get((sentence[-2], sentence[-1], index), fallback)
				prob = l1 * (10**unigram_lp) + l2 * (10**bigram_lp) + l3 * (10**trigram_lp)
			prob_distribution[index] = prob
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