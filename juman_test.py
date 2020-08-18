# from cffi.backend_ctypes import xrange
from pyknp import Juman
import numpy as np
import operator
from importlib import import_module
import MeCab
import string

#
# l = [10, 22, 8, 8, 11]
# print(np.argmax(l))
# print(np.argmin(l))
#
# a = [8, 9, -10, 5, 18, 9]
# print(max(xrange(len(a)), key=lambda x: a[x]))
#
# index, value = max(enumerate(a), key=operator.itemgetter(1))
# print(index, value)

mecab  = MeCab.Tagger()
juman = Juman()
# result = juman.analysis(u"笑顔とフラッシュがやかましいこれが私の自称姉です。")
in_str = u"俺は行かない。"
result = juman.analysis(in_str)
print(','.join(mrph.midasi for mrph in result))

foo = [[] for i in range(7)]
#all the element lists have the same address, hence they are the same listjust shown seven times.
# foo = [[]] * 7　
idxs = [0]
words = ['apple', 'pen', 'pineapple']
for w in words:
    for i in idxs:
        foo[i].append(w)
        print(foo)



def main():
    for mrph in result.mrph_list():
        print(u"%s, \n読み:%s, 原形:%s, 品詞:%s, 品詞細分類:%s, 活用型:%s, 活用形:%s, 意味情報:%s, 代表表記:%s ;"
        % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))
    print()
    for a in mecab.parse(in_str).split('\n'):
        # print(a)
        if a.strip() != 'EOS' and a!='' and a!='\n':
            print(a.split('\t')[1].split(',')[1])



class Dog:
	def playtime(self):
		print('starting import')
		s = import_module('os')
		print('import successful')


def main2():
	tommy = Dog()
	tommy.playtime()

if __name__ == '__main__':
    main()
