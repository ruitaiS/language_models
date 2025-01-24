import data
import random

unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
xft, tfx = data.get_lookups() # index from token, token from index

l1, l2, l3 = (9.49650121e-01, 9.37157253e-18, 5.03498791e-02)

# TODO what is the deal with the janky log conversions back and forth
def output(max_tokens=20):
	prob_distribution = {}
	sentence = [xft['<s>']]
	while len(sentence) < max_tokens:
		for index in tfx.keys():
			if len(sentence) == 1:
				prob = l1 * (10**unigram[index]) + l2 * (10**bigram.get((xft['<s>'], index), -100))
			else:
				prob = l1 * (10**unigram[index]) + l2 * (10**bigram.get((sentence[-1], index), -100)) + l3 * (10**bigram.get((sentence[-2], sentence[-1], index), -100))
			prob_distribution[index] = prob
		#print(sum(prob_distribution.values()))

		# TODO: This sampling method is psychotic and mathematically unsound
		leftover = random.uniform(0, sum(prob_distribution.values()))
		for index, prob in prob_distribution.items():
			leftover -= prob
			if leftover <= 0 :
				sentence.append(index)
				print(sentence)
				break

	return sentence
			



index_list = output()
readable_list = [tfx[index] for index in index_list]
print(" ".join(readable_list))