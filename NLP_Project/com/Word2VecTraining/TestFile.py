# 数据分析
import datetime

print('Hello World!')
print('Time is ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %A'))
print('__name__ value: ', __name__)


def main():
    print('this message is from main function')






def test_funcation():
    list = ['火影', '忍者']
    result = []



    vocab = [(w, i) for i, w in enumerate(list)]
    reverse_vocab = [(i, w) for i, w in enumerate(list)]
    print(vocab)
    print(reverse_vocab)

if __name__ == '__main__':
    main()
    test_funcation()