import os
import random
import time

base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, '../datasets/akjv.txt')
 
def remove_verse_reference(dataset):
    output = []
    for line in dataset:
        parts = line.split("\t")
        if len(parts) > 1:
            text = parts[1].strip()
            output.append(text)
        else:
            print(f"Skipped line: {line}")
    return output

with open(input_file, 'r') as infile:

    # Training / Dev / Test Split
    props = (8, 1, 1) # (Train, Dev, Test) normalized
    props = tuple(p / sum(props) for p in props)

    lines = infile.readlines()
    lines = remove_verse_reference(lines)
    random.shuffle(lines)
    lines = [line + '\n' for line in lines]

    train_set = lines[:int(len(lines)*props[0])]
    dev_set = lines[int(len(lines)*props[0]):int(len(lines)*props[0]) + int(len(lines)*props[1])]
    test_set = lines[int(len(lines)*props[0]) + int(len(lines)*props[1]):]

    time_code = int(time.time())
    fp1 = os.path.join(base_path, f'text/{time_code}/train_set.txt')
    fp2 = os.path.join(base_path, f'text/{time_code}/dev_set.txt')
    fp3 = os.path.join(base_path, f'text/{time_code}/test_set.txt')
    os.makedirs(os.path.dirname(fp1), exist_ok=True)
    os.makedirs(os.path.dirname(fp2), exist_ok=True)
    os.makedirs(os.path.dirname(fp3), exist_ok=True)

    with open(fp1, 'w') as train_file:
        print(f"{len(train_set)} lines in training set.")
        train_file.writelines(train_set)
    
    with open(fp2, "w") as dev_file:
        print(f"{len(dev_set)} lines in dev set.")
        dev_file.writelines(dev_set)

    with open(fp3, 'w') as test_file:
        print(f"{len(test_set)} lines in test set.")
        test_file.writelines(test_set)
