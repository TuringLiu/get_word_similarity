import jieba
import fasttext
import pandas as pd
import numpy as np

# 划分数据集：训练集、验证集
def split_dataset():
    data_path = './cooking.stackexchange.txt'
    data = []
    # 读入数据集
    with open(data_path, 'r') as file:
        for line in file.readlines():
            data.append(line)
    # 划分数据集
    train_data = data[:len(data) - 3000]
    valid_data = data[len(data) - 3000:]

    # print(len(data))
    # 保存训练集
    with open('cooking.train', 'w') as file:
        for d in train_data:
            file.write(d)
    # 保存验证集
    with open('cooking.valid', 'w') as file:
        for d in valid_data:
            file.write(d)

# 模型训练
def model_train():
    model = fasttext.train_supervised(input='cooking.train')
    model.save_model('model_cooking.bin')
    return model

# 文本类别预测
def model_predict(model):
    text = "Which baking dish is best to bake a banana bread ?"
    text_tag = model.predict(text, k=5) # k为预测的类别数
    print(text_tag)

# 模型验证
def model_valid(model):
    test_result = model.test('cooking.valid')
    print(test_result)

if __name__ == '__main__':
    split_dataset()
    model = model_train()
    model_predict(model)
    model_valid(model)
