import os

source_file = "title_list.txt"
target_file = "encoding_title_list.txt"
dict_file = "dict.txt"

with open(dict_file, "r", encoding="utf-8") as f_dict:
    dict_txt = eval(f_dict.readlines()[0])


def line_encoding(line, dict_txt, lable):
    new_line = ""  # 编码后保存的字符串
    for word in line:
        if word in dict_txt:  # 字存在于字典中
            code = str(dict_txt[word])  # 取得编码并转换成字符串
        else:  # 字没有在字典中
            code = str(dict_txt["<unk>"])  # 编码成未知字符
        new_line = new_line + code + ","  # 每个字用逗号分隔
    new_line = new_line[:-1]  # 去掉最后一个多余的逗号
    new_line = new_line + "\t" + lable + "\n"  # 与标记合并成新行
    return new_line


new_line_list = []
with open(source_file, "r", encoding="utf-8") as stf:
    lines = stf.readlines()
    for line in lines:
        words = line.replace("\n", "").split("\t")
        lable = words[1]
        article = words[0]
        new_line = line_encoding(article, dict_txt, lable)
        new_line_list.append(new_line)
with open(target_file, "w", encoding="utf-8") as f_new:
    for oneLine in new_line_list:
        f_new.write(oneLine)
    f_new.close()
print("f_new set is ok")
