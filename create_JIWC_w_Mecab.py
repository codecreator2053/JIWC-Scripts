import pandas as pd
from os import listdir
from os.path import isfile, join, dirname
import numpy as np
import MeCab
import pickle as pkl
from collections import Counter
import string


feels = ['悲しい', '不安', '怒り', '嫌悪感', '信頼感', '驚き', '楽しい']
pos = ['名詞', '動詞', '形容詞', '副詞']
files_dir = "/Users/vipulmishra/Documents/RA work/8月/JIWC/2018_Life_Stories/"
total = '合計'
freq_threshold = 10
katakana = [chr(i) for i in range(12449, 12533)]
hiragana = [chr(i) for i in range(12353, 12436)]
with open("stopwords_jp.txt", "r") as f:
    STOPWORDS_JP = [line.strip() for line in f.readlines()]
STOPPOS_JP = ["形容動詞語幹", "副詞可能", "代名詞", "ナイ形容詞語幹", "特殊", "数", "接尾", "非自立"]
digits = list(string.digits)
Prohibited = STOPWORDS_JP + katakana + hiragana + digits
mecab = MeCab.Tagger()


def get_dummy_rows(df):
    '''get the episode rows which have the same initial string as the col name'''
    tricky_idx = []
    n = 0
    for i in df.iterrows():
        vals = i[1][2:].tolist()
        a = [1 if pd.notna(vals[j]) and type(vals[j]) == str and feels[j] == vals[j][:len(feels[j])]
             else 0 for j in range(len(feels))]
        if np.array(a).sum(0) > 2:
            tricky_idx.append(n)
        n += 1
    print(tricky_idx)


def get_all_words(word_dict, ep_dict):
    '''get all unique words in the episode bank'''
    print('Prohibited', len(Prohibited))
    all_words = []
    freq_above_10 = {}

    for categ, eps in ep_dict.items():
        for ep in eps:
            if pd.notna(ep) and type(ep) == str:
                result = []
                for a in mecab.parse(ep).split('\n'):
                    if a!= 'EOS' and a!='' and a!='\n':
                        result.append([a.split('\t')[0], a.split('\t')[1].split(',')[0], a.split('\t')[1].split(',')[1]])
                all_words.extend([a[0] for a in result if a[1] in pos and a[0] not in Prohibited and a[2] not in STOPPOS_JP])
                word_dict[categ].extend([a[0] for a in result if a[1] in pos and a[0] not in Prohibited and a[2] not in STOPPOS_JP])
        word_dict[categ] = Counter(word_dict[categ])

    all_words = Counter(all_words)
    for i, j in all_words.items():
        if j > freq_threshold:
            freq_above_10[i] = j
    print(len(freq_above_10))
    word_count_feels = {i: {j: None for j in feels} for i in list(freq_above_10.keys())}
    for i in word_count_feels.keys():
        word_count_feels[i][total] = freq_above_10[i]
        for j in feels:
            word_count_feels[i][j] = word_dict[j][i]

    '''better to save in dicts and load when data size is very large'''
    # save_dict(freq_above_10, 'freq_above_10')
    save_dict(word_count_feels, 'word_count_feels')
    # save_dict(word_dict, 'word_count_dict')
    # save_dict(all_words, 'all_words')
    return word_count_feels


def print_elem(no, categ=None):
    '''print element in the ep. bank dict.'''
    if categ is None:
        if no < 4000:
            print(no, [ep_dict[feels[a]][no] for a in range(len(feels))])
        else:
            print(no//4000)
            print(no, ep_dict[feels[no//4000]][no % 4000])
    else:
        print(ep_dict[feels[categ]][no])


def save_dict(dict, name):
    with open(join(dirname(files_dir), name + ".pkl"), "wb") as f:
        pkl.dump(dict, f)


def load_dict(name):
    dic = None
    with open(join(dirname(files_dir), name + ".pkl"), 'rb') as f:
        dic = (pkl.load(f))
    return dic


def get_all_tf_idf(word_count_feels):
    '''preparing a dictionary to store tf-idf values for all words in all categorires'''
    tf_idf_dict = {i: {j: 0.0 for j in feels} for i in word_count_feels.keys()}

    for i in tf_idf_dict.keys():
        for j in tf_idf_dict[i].keys():
            tf = word_count_feels[i][j]
            df = word_count_feels[i][total]
            if df != 0:
                idf = np.log2(tf + 1)/df
                tf_idf = tf * idf
                tf_idf_dict[i][j] = tf_idf
    return tf_idf_dict


def write_tf_file(tf_idf_dict):
    '''to write the JIWC score dataframe to a csv file'''
    tf_alt_format = {i: [tf_idf_dict[a][i] for a in tf_idf_dict.keys()] for i in feels}
    tf_alt_format['単語'] = list(tf_idf_dict.keys())
    df = pd.DataFrame(tf_alt_format)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    print(df.head(5))
    df.to_csv(join(files_dir, 'JIWC_dict-A.csv'), float_format='%.4f')
    # print(tf_alt_format['単語'])


def create_JIWC_dict(tf_idf_dict):
    '''create JIWC dictionary and write to csv file'''
    # print(len(tf_idf_dict.keys()))
    jiwc_dict = [[] for i in range(len(feels))]
    jiwc_dict_b = {a: [] for a in tf_idf_dict.keys()}
    for word, j in tf_idf_dict.items():
        scores = [j[a] for a in feels]
        indices = [a for a in range(len(scores)) if scores[a] == max(scores)]
        for idx in indices:
            jiwc_dict[idx].append(word)
            jiwc_dict_b[word].append(feels[idx])
    jiwc_dict_alt = {feels[a]: pd.Series(jiwc_dict[a]) for a in range(len(feels))}
    jiwc_df = pd.DataFrame(jiwc_dict_alt)
    jiwc_df.to_csv(join(files_dir, 'JIWC_dict-C.csv'), index=False)
    jiwc_df_b = pd.DataFrame.from_dict(jiwc_dict_b, orient='index')
    jiwc_df_b.to_csv(join(files_dir, 'JIWC_dict-B.csv'))


def main():
    ''' get excel files in the folder '''
    dir_content = listdir(files_dir)
    files = [join(files_dir, f) for f in dir_content if isfile(join(files_dir, f)) and f[-4:] == 'xlsx']

    '''read and combine excel files to a dataframe and then to a dict'''
    '''remove excel file other than data file from folder and then run'''
    '''then move other excel files if any back to the folder and use episode_dict.pkl'''
    df1 = pd.read_excel(files[0], 0)
    for f in files[1:]:
        df1 = df1.append(pd.read_excel(f, 0))
    global ep_dict
    ep_dict = {}
    for k in feels:
        ep_dict[k] = [a.strip().replace(' ', '　') if type(a) is str else a for a in df1[k].tolist()]
    save_dict(ep_dict, 'episode_dict')

    # global ep_dict
    # ep_dict = load_dict('episode_dict')
    word_dict = {a: [] for a in feels}

    '''use ths to get dictionary of all words, their total count and count by category'''
    word_count_feels = get_all_words(word_dict, ep_dict)


    # word_count_feels = load_dict('word_count_feels')
    tf_idf_dict = get_all_tf_idf(word_count_feels)
    write_tf_file(tf_idf_dict)
    create_JIWC_dict(tf_idf_dict)


if __name__ == "__main__":
    main()
