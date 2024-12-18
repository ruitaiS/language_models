my_list = [1, 2, 3, 4, 5]

for index, value in enumerate(my_list):
    print(value)
    my_list[index] = value * 2
    print(value)
    #print(f"Index: {index}, Original Value: {value}, Modified Value: {my_list[index]}")

print("Final List:", my_list)