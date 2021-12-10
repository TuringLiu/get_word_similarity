import numpy as np
from gensim.models import word2vec
from gensim import corpora
from gensim import models
from collections import defaultdict
import math

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

def get_texts():
    # 去除常见停用词
    stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # 计算词频
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # 去除仅出现一次的词
    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]
    return texts

# 计算余弦相似度
def get_cos(a, b):
    return (a @ b) / (math.sqrt(np.sum(a ** 2)) * math.sqrt(np.sum(b ** 2)))


# 利用tf-idf表示文本，计算文本相似度
def get_similarity():
    texts = get_texts()
    # for i in texts:
    #     print(i)
    # print(texts)

    # 利用词典生成语料库
    dictionary = corpora.Dictionary(texts)
    # for d in dictionary:
    #     print(d, dictionary[d])
    # 利用语料库，生成稀疏向量表示
    corpus = [dictionary.doc2bow(text) for text in texts]
    # for c in corpus:
    #     print(c)
    # print(corpus)

    tf_idf = models.TfidfModel(corpus)
    # print(tf_idf)

    corpus_tfidf = tf_idf[corpus]
    # for c in corpus_tfidf:
    #     print(c)

    # 利用tf-idf表示，计算文本1和文本4的相似度
    a = np.zeros(12)
    b = np.zeros(12)
    for i in corpus_tfidf[0]:
        a[i[0]] = i[1]
    for i in corpus_tfidf[3]:
        b[i[0]] = i[1]
    a_b_similarity = get_cos(a, b)
    print(a_b_similarity)



def get_lsi():
    texts = get_texts()
    # 利用词典生成语料库
    dictionary = corpora.Dictionary(texts)
    # 利用语料库，生成稀疏向量表示
    corpus = [dictionary.doc2bow(text) for text in texts]
    tf_idf = models.TfidfModel(corpus)
    corpus_tfidf = tf_idf[corpus]
    # 初始化lsi模型
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    # 得到文本的lsi向量值
    corpus_lsi = lsi_model[corpus_tfidf]

    print(lsi_model.print_topics(2))
    for c in corpus_lsi:
        print(c)
    # print(corpus_lsi)




if __name__ == '__main__':
    # gensim_learn()
    # get_similarity()
    get_lsi()
