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

def embedding_layer(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    dictVocab = {}
    vocab_path = './DataFile/vocab.txt'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) == 0:
                continue
            else:
                line = line.strip()
                # print(line)
                line_tmp = line.split("\t")
                dictVocab[line_tmp[0]] = line_tmp[1]
    print(len(dictVocab))
    # [[w1,w2,w3,.......],
    #  [w1,w2,w3,.......],
    #  ]
    # 3 * 4
    # 1-4  1-5
    # 55808 * 256
    # 1 - 55809  1 - 257
    # matrix_one = [[0.0 for j in range(1, 257)] for i in range(1, 55809)]
    # print(matrix_one)
    # for w, v in dictVocab.items():
    #     for word in model.vocab:
    #         if w == word:
    #             print('====line_index====', index)
    #             id = model.vocab[w].index
    #             vec = model.vectors[model.vocab[w].index]
    #             # print(vec)
    #             i = 0
    #             for item in vec:
    #                 matrix_one[int(v)][i] = item
    #                 i = i + 1
    #         else:
    #             continue
    # print("=====print=====a")
    # print(len(matrix_one))
    # print(len(matrix_one[0]))
    # print(matrix_one[0][0])
    # print('词向量维度: ',model.vectors.shape)

# matrix = [[]]
# res = [[]] * len(dictVocab)
# matrix = [list() for i in range(len(dictVocab))]
# a[0].append(1)
# 55809 * 256
# count = 0
# fo = open("embedding_layer.txt", "w")
# for w, v in dictVocab.items():
#     for word in model.vocab:
#         temp = []
#         if w == word:
#             print('====index_id==== ', model.vocab[w].index)
#             vec = model.vectors[model.vocab[w].index]
#             # print(vec)
#             for item in vec:
#                 temp.append(item)
#             # print(temp)
#             # print(len(temp))
#             matrix[int(v)].append(temp)
#             count = count + 1
#         else:
#             # print(int(v))
#             # for i in range(256):
#             #     temp.append(0.0)
#             # matrix[int(v)].append(temp)
#             continue
# print("====print==== matrix")
# print(count)
# print(len(matrix))
# print(matrix[0])
# print('词向量维度: ',model.vectors.shape)

#  5391 * 256
# matrix_two = [list() for i in range(5391)]
# count = 0
# for w, v in dictVocab.items():
#     for word in model.vocab:
#         temp = []
#         if w == word:
#             vec = model.vectors[model.vocab[w].index]
#             # print(vec)
#             for item in vec:
#                 temp.append(item)
#             # print(temp)
#             # print(len(temp))
#             matrix_two[int(v)].append(temp)
#             count = count + 1
#         else:
#             # print(int(v))
#             for i in range(256):
#                 temp.append(0.0)
#             matrix_two[int(v)].append(temp)
#             # continue
# print("====print==== matrix")
# print(count)
# print(len(matrix_two))
# print(matrix_two[0])
print('词向量维度: ', model.vectors.shape)
    '''
        最后修正的emdeding——layer，有待确认
        '''
    matrix_res = [list() for i in range(5391)]
    # a[0].append(1)
    # 55809 * 256
    # count = 1
    # fo = open("embedding_layer_list.txt", "w")
    # for w, v in dictVocab.items():
    #     for word in model.vocab:
    #         temp = []
    #         if w == word:
    #             print('====index_id==== ', model.vocab[w].index)
    #             vec = model.vectors[model.vocab[w].index]
    #             # print(vec)
    #             for item in vec:
    #                 temp.append(item)
    #             # print(temp)
    #             # print(len(temp))
    #             matrix_res[int(model.vocab[w].index)].append(temp)
    #             count = count + 1
    #         else:
    #             # print(int(v))
    #             # for i in range(256):
    #             #     temp.append(0.0)
    #             # matrix[int(v)].append(temp)
    #             continue
    # print("====print==== matrix")
    # print(count)
    # print(len(matrix_res))
    # print(matrix_res[0])
    # print(matrix_res[1])
    # print(matrix_res[3])
    # for temp in matrix_res:
    #     str_res = ""
    #     for tmp in temp:
    #         str_res = str(tmp) + ', '
    #     print("===str=== ")
    #     print(str_res)
    #     # 返回结果是zd：['G20', '放假安排']
    #     fo.write(str_res + '\n')
    # print('词向量维度: ', model.vectors.shape)
    # fo.close()
    fo_one = open("embedding_layer_list_one.txt", "w")
    matrix_one = [[0.0 for j in range(1, 257)] for i in range(1, 5392)]
    print(matrix_one)
    for w, v in dictVocab.items():
        for word in model.vocab:
            if w == word:
                id = model.vocab[w].index
                vec = model.vectors[model.vocab[w].index]
                # print(vec)
                i = 0
                for item in vec:
                    matrix_one[int(id)][i] = item
                    i = i + 1
            else:
                continue
    '''
        embedding_matrix : 对应的训练的index : 和词向量
        分别存入 embedding_layer_list, embedding_layer_list
        '''
print("=====print=====a")
print(len(matrix_one))
print(len(matrix_one[0]))
print(matrix_one[0][0])
for i in range(len(matrix_one)):
    str_res_one = ""
        for j in range(len(matrix_one[0])):
            str_res_one += str(matrix_one[i][j]) + ','
    fo_one.write(str_res_one + '\n')

print('词向量维度: ', model.vectors.shape)
fo_one.close()
print('词向量维度: ',model.vectors.shape)


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))
if __name__ == '__main__':
#    train_build('./DataFile/train_set_seg_x.txt',
#          './DataFile/train_set_seg_y.txt',
#          './DataFile/test_set_seg_x.txt',
#          out_path='./DataFile/word2vec.txt',
#          sentence_path='./DataFile/sentences.txt',)
    embedding_layer('./w2v_sg.bin')
