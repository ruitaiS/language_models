import data

inputs, targets = data.sample(16, 10)

zipped = zip(inputs, targets)
for item in zipped:
	print(item)