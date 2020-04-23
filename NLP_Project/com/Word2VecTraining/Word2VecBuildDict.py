from collections import defaultdict
#   input：处理完的list，save_path:存储的路径
#   output: 字典的存储路径
def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))

def read_data(train_x, train_y, test_x):
    with open(train_x, 'r', encoding='utf-8') as f1, \
        open(train_y, 'r', encoding='utf-8') as f2, \
        open(test_x, 'r', encoding='utf-8') as f3:
        words = []
        for line in f1:
            words = line.split(' ')
        for line in f2:
            words += line.split(' ')
        for line in f3:
            words += line.split(' ')
    return words
"""
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
"""
"""
    词汇的预处理：bug，在遍历list中每一个词的过程，需要做一个预处理
    对于没有完全切好的词，需要做" "判断
"""
def build_vocab(items, sort=True, min_count=0,lower=False):
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # 按照字典中的，每一个词的频率排序
        # 加到list
        # 返回
        dic = sorted(dic.items(), key=lambda d:d[1],reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]
    return vocab, reverse_vocab

if __name__ == '__main__':
    # add all text data
    # 加入之前预处理数据和停用词都不一样，最后得到的结果也不一样
    # 最后的效果，需要迭代
    lines = read_data('./DataFile/train_set_seg_x.txt',
                      './DataFile/train_set_seg_y.txt',
                      './DataFile/test_set_seg_x.txt')
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, './DataFile/vocab.txt')