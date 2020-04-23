# 数据分析
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import sys
from gensim.models import Word2Vec
import jieba.posseg as pseg
import jieba.analyse
# input trainingPath, testPath
# output - dataFrame

def dataProcessing(traingingDataPath, testDataPath):
    # check training data
    df = pd.read_csv(traingingDataPath, encoding='utf-8')
    # df.info()
    # check data shape
    # df.shape
    # check null values for data
    # df.isnull().sum()
    # 根据业务场景，标签是report，根据问题和对话构建的report，所以对于report来说，没有相关联
    # 所以删掉这两列
    df1 = df.drop('Brand', axis=1)
    df_res = df1.drop('Model', axis=1)
    df_res = df_res.drop('QID', axis=1)
    # df_res.info()
    # check test data
    # Dialogue 有两行为null
    # Report 有70行为null
    # 计算null所占的比例
    # dataFunction(df_res)
    #return traingingDataFrame, testDataFrame
    # 因为是做文本的标签处理，对于report 为NaN的值有，不同的看法
    # method one 可以处理为随时联系 - 主观意识
    # method two 因为标签是给予问题，鉴于数量比较少可以人工标注；标签为文本，存在不确定性或者可以把NaN的删掉
    # method three 可以用其他算法填充，需要讨论
    df_res.dropna(subset=['Report'], how='any', inplace=True)
    df_res.info()
    # df_res.to_csv('./df_res_training_data_step_one.csv', index=False)
    # training_data_one = pd.read_csv('./df_res_training_data_step_one.csv', encoding='utf-8')
    #df_res.info()
    # 剩余的只有 Dialogue，如果有空，
    df_res.fillna('', inplace=True)
    df_res.info()
    # training_data_one.to_csv('./df_res_training_data_step_one.csv', index = False)
    # 对于标签的合并
    df_res.Report.value_counts()
    df_res.loc[:, 'Report'] = df_res['Report'].str.replace("随时联系！", "随时联系")
    print("===== train data ======")
    df_res.info()
    train_x = df_res.Question.str.cat(df_res.Dialogue)
    train_y = []
    if 'Report' in df_res.columns:
        train_y = df_res.Report
        assert len(train_x) == len(train_y)
    # check test data
    print("====== test data=====")
    test_data = pd.read_csv(testDataPath, encoding='utf-8')
    test_data.isnull().sum()
    test_data.info()
    dataFunction(test_data)
    test_data = test_data.drop('Brand', axis=1)
    test_data = test_data.drop('Model', axis=1)
    test_data = test_data.drop('QID', axis=1)
    test_data.info()
    test_x = test_data.Question.str.cat(test_data.Dialogue)
    test_y = []
    return train_x, train_y, test_x, test_y

# null 值的占比计算
dd=[]
cc=[]
rr=[]
# null 值的比例计算
outfile = './NullRatioAna.csv'
data=pd.DataFrame()
# null values ana
def dataFunction(a):
    for i in a.columns:
        d=len(a)-a[i].count()
        r=(d/len(a))*100
        rate='%.2f%%' % r
        print('字段名为：',str(i).ljust(10),'缺失值数量:',str(d).ljust(4),'缺失数量占比：',rate) #这里print主要是为了在脚本中观察是否获取到想要的数据，方便调试。
        dd.append(i)
        cc.append(d)
        rr.append(rate)
    data[u'字段名为']=dd
    data[u'缺失值数量']=cc
    data[u'缺失数量占比']=rr
    outfile=r'./NullRatioAna.xls'
    data.to_excel(outfile) #同样输出路径尽量用英文，输出为xls格式



if __name__ == '__main__':
    traingDataPath = "/Users/mac/Desktop/NLP_Project/com/Word2VecTraining/AutoMaster_TrainSet.csv"
    testDataPath = "/Users/mac/Desktop/NLP_Project/com/Word2VecTraining/AutoMaster_TestSet.csv"
    dataProcessing(traingDataPath, testDataPath)
