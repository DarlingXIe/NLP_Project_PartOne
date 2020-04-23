from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
'''
    存储word的字典
'''
def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))
'''
    读取数据: 每一个文件夹下的数据
'''
def read_lines(path, col_seq=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_seq:
                if col_seq in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines
'''
    读取数据: 拼接三个路径下的text数据
'''
def all_data_sentence(train_set_seg_x, train_set_seg_y, test_set_seg_x):
    ret = []
    lines = read_lines(train_set_seg_x)
    lines += read_lines(train_set_seg_y)
    lines += read_lines(test_set_seg_x)
    for line in lines:
        ret.append(line)
    return ret
'''
    读取数据: 保存所有拼接数据
'''
def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('保存句子:%s' % sentence_path)

'''
   读取数据, 训练数据
   训练模型
'''
def train_build(train_set_seg_x_path, train_set_seg_y_path, test_set_seg_x, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=100):
    sentences = all_data_sentence(train_set_seg_x_path, train_set_seg_y_path, test_set_seg_x)
    # 保存所有text数据
    save_sentence(sentences, sentence_path)
    print('.....训练模型.....')
    w2v = Word2Vec(sg=0, sentences=LineSentence(sentence_path),
                   size=256, window=10, min_count=min_count, iter=5)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('检查', '车主')
    print('技师 vs 车主 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    #dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    train_build('./DataFile/train_set_seg_x.txt',
          './DataFile/train_set_seg_y.txt',
          './DataFile/test_set_seg_x.txt',
          out_path='./DataFile/word2vec.txt',
          sentence_path='./DataFile/sentences.txt',)