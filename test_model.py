# 模型加载、预测
# 将句子进行编码
import numpy as np
from multiprocessing import cpu_count
import os
import paddle
import paddle.fluid as fluid

dict_file_path = "data/dict.txt"


def get_data(sentence):
    with open(dict_file_path, "r", encoding="utf-8") as f:
        dict_txt = eval(f.readlines()[0])
    ret = []  # 返回值：经过编码转换的列表
    for s in sentence:
        if not s in dict_txt.keys():  # 字没有在字典中
            s = "<unk>"
        ret.append(int(dict_txt[s]))  # 找到字的编码并放入ret列表中
    return ret


# 加载模型
model_save_dir = "model/news_classify"
place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

infer_program, feeded_var_names, target_var = \
    fluid.io.load_inference_model(dirname=model_save_dir, executor=exe)
print("加载模型完成")


texts = []  # 预测句子的列表
data1 = get_data("香港牛年表现最佳五大国企股")
data2 = get_data("京东商城中标家电以旧换新，用户足不出户获补贴")
data3 = get_data("中国队无缘2020年世界杯")
data4 = get_data("10月20日，第六届世界互联网大会正式开幕")

texts.append(data1)
texts.append(data2)
texts.append(data3)
texts.append(data4)
base_shape = [[len(c) for c in texts]]  # 获取每个句子的长度并且放入数组中

# 将经过编码后的句子转换为张量
tensor_words = fluid.create_lod_tensor(texts, base_shape, place)
# 源数据，数据长度，，

# 执行预测
result = exe.run(infer_program, feed={
                 feeded_var_names[0]: tensor_words}, fetch_list=target_var)
# 预测program，喂入参数，获取结果

names = ["社会", "时尚", "科技", "财经", "房产", "彩票", "娱乐",
         "股票", "教育", "游戏", "星座", "时政", "体育", "家居"]
# 获取结果概率最大的label
for i in range(len(texts)):
    lab = np.argsort(result)[0][i][-1]
    print("预测结果:%d, 名称:%s, 概率:%f" % (lab, names[lab], result[0][i][lab]))
