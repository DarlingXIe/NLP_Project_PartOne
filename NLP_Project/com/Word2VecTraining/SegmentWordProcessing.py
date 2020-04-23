import jieba
from jieba import posseg
import sys
sys.path.append("./DataProcessing")
from DataProcessing import dataProcessing
#
#  数据来说 - 查看数据中的问题
# 【图片]| 有一些类似于这样数据格式，不需要，[语音]
#
REMOVE_WORDS = ['|', '[', ']', '语音', '图片']
#
#  remove some words
#  input : word list
#  output : word list
def remove_words(words_lists):
    words_lists = [word for word in words_lists if word not in REMOVE_WORDS]
    return words_lists
#
#  remove stops word:
#  结合业务场景，不是所用的停用词，都是可以去掉的，需要修改停用词的列表
#  用最全停用词表
#
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 普通的分词返回str
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip(),cut_all = False)
    stopwords = stopwordslist('./stop_words.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
#
#   input ：句子 sentence; 句子中词的类型 word_type = 'word', 'char'; 是否分词: pos = true, pos = false
#   output: 词的字典；字典的类型
#
def seq_sentence(sentence, cut_type = 'word', pos = False):
    if pos:
        # dict = {}
        if cut_type == 'word':
            word_pos_neg_sentence = posseg.lcut(sentence.strip())
            word_sql,pos_neg = [],[]
            for w, p in word_pos_neg_sentence:
                word_sql.append(w)
                pos_neg.append(p)
            return word_sql, pos_neg
        elif cut_type == 'char':
            word_sql = list(sentence)
            for word in word_sql:
                word_seq = posseg.lcut(word)
                pos_neg = []
                for w in word_sql:
                    pos_neg.append(w[0].flag)
            return word_seq, pos_neg
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence.strip(), cut_all = False)
        elif cut_type == 'char':
            return list(sentence)

#
#   数据的保存：input : train_x, train_y, test_x, test_y, train_x_path, train_y_path, test_x_path, stop_words_path
#             output : train_y_path, train_y_path, test_x_path
#
def save_train_test_data(train_x, train_y, test_x, test_y,train_x_path, train_y_path,test_x_path, stop_words_path):
    stopwords = stopwordslist(stop_words_path)
    # with open(train_x_path, 'w', encoding='utf-8') as w_1:
    #     count_train_x = 0
    #     for line in train_x:
    #         if isinstance(line, str):
    #             seg_list = seq_sentence(line, 'word')
    #             temp_res = remove_words(seg_list)
    #             res = [word for word in temp_res if word not in stopwords]
    #             if (len(res)) > 0:
    #                 res_line = ' '.join(res)
    #                 w_1.write('%s' % res_line)
    #                 w_1.write('\n')
    #                 count_train_x += 1
    #     print('train_x_length is ', count_train_x)

    # with open(train_y_path, 'w', encoding='utf-8') as w_2:
    #     count_train_y = 0
    #     for line in train_y:
    #         if isinstance(line, str):
    #             seg_list = seq_sentence(line, 'word')
    #             temp_res = remove_words(seg_list)
    #             res = [word for word in temp_res if word not in stopwords]
    #             if (len(res)) > 0:
    #                 res_line = ' '.join(res)
    #                 w_2.write('%s' % res_line)
    #                 w_2.write('\n')
    #             else:
    #                 w_2.write("随时联系")
    #                 w_2.write('\n')
    #             count_train_y += 1
    #     print('train_y_length is ', count_train_y)

    with open(test_x_path, 'w', encoding='utf-8') as w_3:
        count_test_x = 0
        for line in test_x:
            if isinstance(line, str):
                seg_list = seq_sentence(line, 'word')
                temp_res = remove_words(seg_list)
                res = [word for word in temp_res if word not in stopwords]
                if (len(res)) > 0:
                    res_line = ' '.join(res)
                    w_3.write('%s' % res_line)
                    w_3.write('\n')
                    count_test_x += 1
        print('count_test_x is ', count_test_x)

if __name__ == '__main__':
    traingDataPath = "/Users/mac/Desktop/NLP_Project/com/Word2VecTraining/AutoMaster_TrainSet.csv"
    testDataPath = "/Users/mac/Desktop/NLP_Project/com/Word2VecTraining/AutoMaster_TestSet.csv"
    train_x, train_y, test_x, _ = dataProcessing(traingDataPath, testDataPath)
    train_x_path = "./DataFile/train_set_seg_x.txt"
    train_y_path = "./DataFile/train_set_seg_y.txt"
    test_x_path = "./DataFile/test_set_seg_x.txt"
    test_y_path = ""
    stop_word_path = "./stop_words.txt"
    save_train_test_data(train_x,
                         train_y,
                         test_x,
                         _,
                         train_x_path,
                         train_y_path,
                         test_x_path,
                         stop_word_path)