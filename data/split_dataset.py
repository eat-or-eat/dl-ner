import os
import re

# 生成数据集文件夹
dirs = ['./train/', './test/']
for dir in dirs:
    if not os.path.isdir(dir):
        os.mkdir(dir)

files = ['./source_BIO_2014_cropus.txt', './target_BIO_2014_cropus.txt']
split_num = [20000, 5000]
for file in files:
    data = open(file, 'r', encoding='utf8').readlines()
    info_type = ''.join(re.findall('(?<=/).*?(?=_)', file))  # 提取source或target
    with open(dirs[0] + info_type + '.txt', 'w', encoding='utf8') as f:
        for i in range(split_num[0]):
            f.write(data[i])
    with open(dirs[1] + info_type + '.txt', 'w', encoding='utf8') as f:
        for i in range(split_num[1]):
            f.write(data[split_num[0] + i])

