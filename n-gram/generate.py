import data
import random

unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
xft, tfx = data.get_lookups()

l1, l2, l3 = (0.11669758, 0.50255293, 0.38074949)
fallback = -100

def output(max_tokens=200):
	sentence = [xft['<s>']]
	# prob_distribution = {} # TODO: figure out why this works
	while len(sentence) < max_tokens:
		prob_distribution = {} # TODO: when really it should be here instead
		for index in tfx.keys(): # Iterate over all possible next words
			if len(sentence) == 1:
				bigram_lp = bigram.get((xft['<s>'], index), fallback)
				prob = 10**bigram_lp
				'''
				Really this is count of (xft[<s>], index) bigrams over count of all (<s>, *) bigrams
				(<s>, *) bigrams also equal to count <s> unigrams
				summed over vocab will give total prob of 1
				'''
			else:
				'''
				Summed over vocab, each of uni, bi, tri will contribute l1, l2, l3 proportion of 1

				On special characters, you need to inflate l2 and l3 to accomodate dropping out l1
				Alternatively you need to tune for p2 and p3 params specifically for when l1 is ignored

				uni_prob = unigram_counts(index) / sum(unigram_counts.values())

				bigram = (sentence[-1], index)
				bigram_prob = bigram_counts(bigram) / unigram_count(bigram[0])

				trigram = (sentence[-2], sentence[-1], index)
				trigram_prob = trigram_counts(trigram) / bigram_count((trigram[0], trigram[1]))

				'''
				unigram_lp = unigram.get(index, fallback)
				bigram_lp = bigram.get((sentence[-1], index), fallback)
				trigram_lp = trigram.get((sentence[-2], sentence[-1], index), fallback)
				if int(index) <= 10 or int(index) == 9833: # Ignore unigram probs for punctuation and 's
					prob = (l2 * (10**bigram_lp) + l3 * (10**trigram_lp))/(1-l1)
				else:
					prob = l1 * (10**unigram_lp) + l2 * (10**bigram_lp) + l3 * (10**trigram_lp)
			prob_distribution[index] = prob ## TODO: Verify super janky softmax

		if len(sentence) == 1:
			print(sum(prob_distribution.values()))
		if len(sentence) == 2:
			print(sum(prob_distribution.values()))
		if len(sentence) == 3:
			print(sum(prob_distribution.values()))
		if len(sentence) == 4:
			print(sum(prob_distribution.values()))
		if len(sentence) == 5:
			print(sum(prob_distribution.values()))
		if len(sentence) == 6:
			print(sum(prob_distribution.values()))

		# TODO: This sampling method is psychotic and mathematically unsound
		leftover = random.uniform(0, sum(prob_distribution.values()))
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

'''sentences = []
for i in range(10):
	index_list = output()
	readable_list = [tfx[index] for index in index_list]
	sentences.append(" ".join(readable_list))
print(sentences)# for sentence in sentences'''