# 机器学习大作业 新闻文本分类

## 1.数据搜寻和预处理

### (1) 数据搜寻

对于新闻包括标题和正文部分的数据集，来源于对于近五年的新闻的整合，早在2010年左右，当时的一些主流媒体已经采用对于新闻进行网络分类展示，常见的类别包括但不限于经济、文化、娱乐、教育、军事等，值得庆幸的是，近十年内出现在腾讯、搜狐、头条新闻等媒体的常规分类和以往并无差别，这也就为我们数据集选择并应用于现在的互联网环境提供了现实基础，不至于构建出一个由于分类多变导致不稳定学习模型。

我们对于数据的选择并不是将网络上的信息照单全收，而是有选择采用，比如：

①尽量采用较新的新闻文本，考虑到一些特殊情况，如将2020年之前新闻的大量采用对于现在社会疫情环境下的新闻分类显然会产生问题，因为在2020年的新闻标题正文完全没有“新冠疫情”这类词汇，而对于采用2020年之前数据训练出的模型使用当今疫情环境下的新闻标题则会产生较大的偏差，就像将“社会”归类于“医疗”的错误。

②对于一些分类的舍去，比如一些不符合现状的分类和极其重复的分类

③对于新闻文本标题和正文的综合考量，现代互联网新闻的特性决定我们不能单一的考虑标题或者新闻正文，很明显，对于标题的训练不论是效率、代价还是实际应用都优于正文，但由于当今信息碎片化的需求，新闻标题往往会为了博人眼球而“剑走偏锋”，俗称“标题党”，如此一来对于正文的文本信息获取也是必不可少的辅助方式。

我们团队采取的主要方式是使用python爬虫来获取搜狐、腾讯的新闻信息，包括分析目标网站，包括 URL 结构，HTML 页面，网络请求及返回的结果等，找到我们要爬取的目标位于网站的哪一个位置；接下来，使用代码去发起网络请求，解析网页内容，提取目标数据，并进行数据存储；最后，测试代码，完善程序。文件形式如下图：


```python
import requests
from bs4 import BeautifulSoup
import time
import json
import re
import pandas
import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')
def getnewcontent(url):
    result = {}
    info = requests.get(url)
    info.encoding = 'utf-8'
    html = BeautifulSoup(info.text, 'html.parser')
    result['title'] = html.select('.second-title')[0].text
    result['date'] = html.select('.date')[0].text
    result['source'] = html.select('.source')[0].text
    article = []
    for v in html.select('.article p')[:-1]:
        article.append(v.text.strip())
    author_info = '\n'.join(article)
    result['content'] = author_info
    result['author'] = html.select('.show_author')[0].text.lstrip('责任编辑：')
    newsid = url.split('/')[-1].rstrip('.shtml').lstrip('doc-i')
    commenturl = 'http://comment5.news.sina.com.cn/page/info?version=1&format=json&channel=gj&newsid=comos-{}&group=undefined&compress=0&ie=utf-8&oe=utf-8&page=1&page_size=3&t_size=3&h_size=3&thread=1&callback=jsonp_1536041889769&_=1536041889769'
    comments = requests.get(commenturl.format(newsid))
    regex = re.compile(r'(.*?)\(')#去除左边特殊符号
    tmp = comments.text.lstrip(regex.search(comments.text).group())
    jd = json.loads(tmp.rstrip(')'))
    result['comment'] = jd['result']['count']['total'] #获取评论数
    return result
def getnewslink(url):
    test = requests.get(url)
    test2 =  test.text.lstrip('newsloadercallback(')
    jd = json.loads(test2.rstrip(')\n'))
    content = []
    for v in jd['result']['data']:
        content.append(getnewcontent(v['url']))
    return content
def getdata():
    url = 'https://interface.sina.cn/news/get_news_by_channel_new_v2018.d.html?cat_1=51923&show_num=27&level=1,2&page={}&callback=newsloadercallback&_=1536044408917'
    weibo_info = []
    for i in range(1,3):
        newsurl = url.format(i)#字符串格式化用i替换{}
        weibo_info.extend(getnewslink(newsurl))
    return weibo_info
new_info = getdata()
df = pandas.DataFrame(new_info)
df #去除全部 df.head() 取出5行 head(n)  n行
#将文件下载为excel表格 df.to_excel('weibonews.xlsx')
```

#### 分别获取标题和正文的新闻文本数据
![title](./origin_title_list.png)
![title](./origin_full_list.png)

### (2) 数据预处理

我们的数据在通过爬虫获取后就已经是对于每一条新闻包括序号、分类和内容：

对于数据的预处理包括：

①将汉字编码成对应整数，我们了解了几种编码方式。对于中文文本来说，由于词组数目巨多且难以处理，我们选择将文本中所有的汉字根据文件信息的顺序编码形式，使用一个字典将其存储，构造一个汉字与数字一一对应、不重复的字典。


```python
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
print("已经生成字典文件")

```

    已经生成字典文件
    


```python
# 显示字典的前20个元素
dict_path = "data/dict.txt"
with open(dict_path, "r", encoding="utf-8") as f:
    d = eval(f.readlines()[0])
print(list(d.items())[:20])
```

    [('汨', 0), ('鲨', 1), ('蝎', 2), ('≠', 3), ('策', 4), ('直', 5), ('揉', 6), ('笤', 7), ('Ｆ', 8), ('Ｖ', 9), ('抬', 10), ('破', 11), ('毋', 12), ('浦', 13), ('淡', 14), ('琥', 15), ('婵', 16), ('笔', 17), ('咂', 18), ('腌', 19)]
    

②对标题/正文字符串进行解析，包括分离去换行符等

我们搜寻到的数据新闻文本与类别之间用"\t"分割，同时类别为中文，不方便我们后续处理


```python
# 展示原始数据
with open("./data/title_list.txt", "r", encoding="utf-8") as myfile:
    head = [next(myfile) for x in range(10)]
print(head)
```

    ['网易第三季度业绩低于分析师预期\t科技\n', '巴萨1年前地狱重现这次却是天堂，再赴魔鬼客场必翻盘\t体育\n', '美国称支持向朝鲜提供紧急人道主义援助\t时政\n', '增资交银康联，交行夺参股险商首单\t股票\n', '午盘：原材料板块领涨大盘\t股票\n', '夏日大学游园会，诺基亚E66红黑独家对比\t科技\n', '蔡少芬要补交税款几十万，圣诞节拼命赚外快(图)\t娱乐\n', '盛大文学：挖掘文学的2.0价值\t科技\n', '马队向圣西罗看台竖中指？，告别夜他被谁激怒(图)\t体育\n', '湖人不理性续约渐露恶果，中产神射手一赛季变铁匠\t体育\n']
    

因此我们要将后面的类别数据替换为数字以便后续分类使用


```python
import os
label_file_path = "./data/label_list.txt"
title_list_path = "./data/title_list.txt"
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
        print(label_str)
print("已经完成")
```

    财经	0
    
    体育	1
    
    时政	2
    
    教育	3
    
    游戏	4
    
    娱乐	5
    
    科技	6
    
    彩票	7
    
    房产	8
    
    时尚	9
    
    家居	10
    
    社会	11
    
    股票	12
    
    星座	13
    
    已经完成
    


```python
# 展示label_list
with open("./data/label_list.txt", "r", encoding="utf-8") as myfile:
    head = [next(myfile) for x in range(10)]
print(head)
```

    ['社会\t0\n', '时尚\t1\n', '科技\t2\n', '财经\t3\n', '房产\t4\n', '彩票\t5\n', '娱乐\t6\n', '股票\t7\n', '教育\t8\n', '游戏\t9\n']
    

然后根据label_list信息修改原始文件，将中文类别替换为对应的数字

③将文字标题对照字典进行转换，并标记类别


```python
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

```

    f_new set is ok
    

在此我们得到了对照字典编码过的文本信息，文本信息中每个汉字的编码之间用英文逗号分割，最后使用"\t"分割类别


```python
# 显示编码后的文件,显示前十行的数据
with open("./data/encoding_title_list.txt", "r", encoding="utf-8") as myfile:
    for i in range(10):
        print(next(myfile))
```

    1847,3440,2912,1277,1233,1899,4172,4344,656,1895,4285,1341,965,1650,2483	2
    
    2968,4169,727,500,3874,5091,2048,3291,709,1797,324,1314,5028,2589,1236,3055,2584,2980,3453,3177,2199,1175,4388,3831,2643	12
    
    2855,53,867,464,4286,3679,2148,85,3877,703,2689,4088,4291,2405,3754,1943,2357,590	11
    
    2941,4311,450,1165,503,3084,3055,450,1073,4465,2276,810,3571,1009,3292,3476	7
    
    1155,2643,1214,4636,4468,3026,4400,5005,2670,523,345,2643	7
    
    2051,306,345,566,2723,155,275,3055,980,1484,2283,1630,2327,2327,4810,3720,2145,775,845,4825	2
    
    1969,4424,5106,1082,119,450,1730,2194,322,3911,3143,3055,3512,4191,2731,5023,461,3540,4498,2507,1437,3640,2487	6
    
    5246,345,3222,566,1214,218,2099,3222,566,3568,1799,2519,434,1824,2806	2
    
    4257,1465,3679,3512,1406,3162,5119,3427,3645,1964,3096,1593,3055,3295,4837,2416,4048,2889,1122,2550,1987,1437,3640,2487	12
    
    2756,4291,359,3436,4259,4966,2553,5067,2664,4752,759,3055,1964,631,1238,34,2162,1029,1757,1233,2873,3925,5035	12
    
    

④将文件分测试集、训练集，我们将90%用于训练集，将10%用于测试集，最终将其分为train_list和test_list两个文件


```python
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

```

得到了文件"train.txt"和"test.txt"。


```python
train_path = "./data/train.txt"
test_path = "./data/test.txt"

with open(train_path, "r", encoding="utf-8") as sf:
    lines = sf.readlines()
print("训练集",len(lines))
with open(test_path, "r", encoding="utf-8") as sf:
    lines = sf.readlines()
print("测试集",len(lines))
```

    训练集 677223
    测试集 75248
    

通过获取两个文件的大小可以验证训练集:测试集=9:1，满足我们的要求。至此数据预处理完成。
