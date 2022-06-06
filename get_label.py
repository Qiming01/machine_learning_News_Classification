import os
label_file_path = "label_list.txt"
title_list_path = "title_list.txt"
label_set = set()
with open(title_list_path, "r", encoding="utf-8") as titlelf:
    lines = titlelf.readlines()
for line in lines:
    label = line.replace("\n", "").split("\t")[-1]
    label_set.add(label)
i = 0
with open(label_file_path, "w", encoding="utf-8") as labellf:
    for onelabel in label_set:
        label_str = onelabel + "\t" + str(i) + "\n"
        i += 1
        labellf.write(label_str)
