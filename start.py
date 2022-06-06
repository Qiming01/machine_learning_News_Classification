from multiprocessing import cpu_count
import os
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt


dict_path = "data/dict.txt"
train_data_file = "data/train.txt"
test_file_path = "data/test.txt"
paddle.enable_static()
# 获取字典长度


def get_dict_len(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        d = eval(f.readlines()[0])
    return len(d.keys())  # 返回字典对象值的个数


#创建reader(train_reader, test_reader)
def data_mapper(sample):
    data, label = sample  # 将样本数据赋值给data, label
    val = [int(w) for w in data.split(",")]  # 将样本以“，”分割成一个列表
    return val, int(label)


def train_reader(train_data_file):

    def reader():
        with open(train_data_file, "r") as f:
            lines = f.readlines()
            np.random.shuffle(lines)  # 打乱

            for line in lines:
                data, label = line.split("\t")
                yield data, label

    # 将mapper生成的数据交给reader进行二次处理并输出
    return paddle.reader.xmap_readers(data_mapper,  # reader函数
                                      reader,  # 产生数据的reader
                                      cpu_count(),  # 线程数
                                      1024)  # 缓冲区大小


def test_reader(test_file_path):
    def reader():
        with open(test_file_path, "r") as f:
            lines = f.readlines()
            np.random.shuffle(lines)  # 打乱

            for line in lines:
                data, label = line.split("\t")
                yield data, label

    # 将mapper生成的数据交给reader进行二次处理并输出
    return paddle.reader.xmap_readers(data_mapper,  # reader函数
                                      reader,  # 产生数据的reader
                                      cpu_count(),  # 线程数
                                      1024)  # 缓冲区大小
# 搭建网络
# 嵌入层 --> 卷积/池化层
#     /--> 卷积/池化层 --> 全连接层


def CNN_net(data, dict_dim, class_dim=14,
            emb_dim=128, hid_dim=128, hid_dim2=98):
    # embding（词向量层）：将高度系数的离散输入嵌入到一个新的实向量空间
    # 将稀疏举证表示为稠密序列矩阵
    # 用更少的维数，表示更多的信息
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # 第一个卷积/池化层
    conv_1 = fluid.nets.sequence_conv_pool(
        input=emb, num_filters=hid_dim, filter_size=3, act="tanh", pool_type="sqrt")
    # 输入，卷积核数，卷积核大小，激活函数，池化类型
    conv_2 = fluid.nets.sequence_conv_pool(
        input=emb, num_filters=hid_dim2, filter_size=4, act="tanh", pool_type="sqrt")
    output = fluid.layers.fc(
        input=[conv_1, conv_2], size=class_dim, act="softmax")
    # 输入，输出类别个数：10，激活函数
    return output


# 读取数据，训练模型
EPOCH_NUM = 5  # 训练迭代次数
model_save_dir = "model/news_classify/"  # 模型保存路径
words = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")
# 获取字典长度
dict_dim = get_dict_len(dict_path)
# 生成神经网络
model = CNN_net(words, dict_dim)
# 定义损失函数：交叉熵
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
# 计算准确率
acc = fluid.layers.accuracy(input=model, label=label)
# 赋值program用于测试
test_program = fluid.default_main_program().clone(for_test=True)
# 定义优化器
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)
opt = optimizer.minimize(avg_cost)  # 求平均损失最小值


# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化

# 准备
tr_reader = train_reader(train_data_file)
train_reader = paddle.batch(reader=tr_reader, batch_size=128)
ts_reader = test_reader(test_file_path)
test_reader = paddle.batch(reader=ts_reader, batch_size=128)

feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

# 定义变量可视化
times = 0
batches = []  # 迭代次数列表
costs = []  # 损失值的列表
accs = []  # 准确率列表

# 开始训练
for pass_id in range(EPOCH_NUM):
    for batch_id, data in enumerate(train_reader()):
        times += 1  # 训练次数加一
        train_cost, train_acc = exe.run(program=fluid.default_main_program(
        ), feed=feeder.feed(data), fetch_list=[avg_cost, acc])
    # 每100笔打印一次准确率和损失值
        if batch_id % 100 == 0:
            print("pass_id:%d, batch_id:%d, cost:%f, acc:%f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
            accs.append(train_acc[0])  # 记录准确率
            costs.append(train_cost[0])  # 记录损失值
            batches.append(times)  # 记录次数

# 模型保存
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
fluid.io.save_inference_model(model_save_dir, feeded_var_names=[
                              words.name], target_vars=[model], executor=exe)
print("模型保存完成！")
