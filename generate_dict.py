title_list_path = "title_list.txt"
dict_file_path = "dict.txt"
dict_set = set()
with open(title_list_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    title = line.split("\t")[0]
    for w in title:
        dict_set.add(w)
dict_list = []
i = 0
for s in dict_set:
    dict_list.append([s, i])
    i += 1
dict_txt = dict(dict_list)
end_dict = {"<unk>": i}
dict_txt.update(end_dict)

with open(dict_file_path, "w", encoding="utf-8") as f:
    f.write(str(dict_txt))
print("ok")
