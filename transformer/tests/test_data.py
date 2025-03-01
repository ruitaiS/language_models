import data

batch_size = 16
context_len = 4

# TODO: finish these up 
def check_chars_by_dataset(dataset_name):
	input_batches, target_batches = data.get_sequences(batch_size, context_len, dataset_name)
	xft, tfx = data.get_vocab()
	assert '<?>' in xft and xft['<?>'] in tfx, '<?> missing from vocab'
	assert '<s>' in xft and xft['<s>'] in tfx, '<s> missing from vocab'
	assert '</s>' in xft and xft['</s>'] in tfx, '</s> missing from vocab'
	assert '<>' in xft and xft['<>'] in tfx, '<> missing from vocab'

	unk_char = xft['<?>']
	start_char = xft['<s>']
	end_char = xft['</s>']
	pad_char = xft['<>']

	# <s> should only ever be at the start of sequences
	# </s> should never be in an input batch; should only ever be in a target batch 
	# <?> should never appear in the training set
	if dataset_name == 'train':
		assert all(unk_char not in seq for seq in input_batches), f'"<?>" should not appear but does, at : {" ".join([tfx[index] for index in seq])} in dataset {dataset_name}'
		assert all(unk_char not in seq for seq in target_batches), f'"<?>" should not appear but does, at : {' '.join([tfx[index] for index in seq])} in dataset {dataset_name}'
	# The number of [<>, <>, ..., <s>] inputs, [...,...,...,</s>] targets, and number of sentences are all equivalent 
	# In inputs, the number of sequences with <> in them will be (context length - 1) * # of sentences.
	# In inputs, the number of sequences with <s> in them should be context_len * (# of sentences)

def test_special_characters():
	for dataset_name in ['train', 'dev', 'test']:
		check_chars_by_dataset(dataset_name)
