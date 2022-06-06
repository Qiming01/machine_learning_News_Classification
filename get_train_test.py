import os

source_path = "encoding_title_list.txt"
train_path = "train.txt"
test_path = "test.txt"

with open(source_path, "r", encoding="utf-8") as sf:
    lines = sf.readlines()

length = len(lines)
i = 0
train_list = []
test_list = []
for line in lines:
    i = i + 1
    if i < 0.9*length:
        train_list.append(line)
    else:
        test_list.append(line)
with open(train_path, "w", encoding="utf-8") as trainf:
    for line in train_list:
        trainf.write(line)

with open(test_path, "w", encoding="utf-8") as testf:
    for line in test_list:
        testf.write(line)
