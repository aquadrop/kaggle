import requests
from gensim.models.wrappers import FastText
from utils.query_util import tokenize
# from utils.embedding.vector_helper import computeSentenceSimWithWeights

import fastText as ft

FASTTEXT_URL = 'http://localhost:11425/fasttext/w2v?q='
FASTTEXT_URL_M = 'http://localhost:11425/fasttext/maxsim?q1={0}&q2={1}'
FASTTEXT_URL_M_POST = 'http://localhost:11425/fasttext/maxsim'
# print('load model')
# model = FastText.load_fasttext_format('/opt/fasttext/model/test.bin')

ft_model = ft.load_model('/media/deep/DATA1/share/fasttext/wiki-news-300d-1M-subword.vec')
def ff_embedding(word):
    vector = ft_model[word]
    return vector

def mlt_ff_embedding(q1, q2):
    # print('q1:',q1)
    # print('q2:',q2)
    # ff_url = FASTTEXT_URL_M.format(q1, q2)
    # print(requests.get(url=ff_url))
    r = requests.post(url=FASTTEXT_URL_M_POST, data={"q1":q1, "q2":q2}).json()
    # print('json:',r)
    sim = r['maxcossim']
    simq = r['simstring']
    return float(sim), simq.replace(',', '')

def ner_weight(pseg_tokens):
    sent1 = {}
    for w, tag in pseg_tokens:
        sent1[w] = 1.0
        if tag in ["n", "nr", "ns", "nt", "nz"]:
            sent1[w] = 2.0
        if tag in ["m"]:
            sent1[w] = 0.5
    return sent1

def w2v_local_similarity(query1, query2s):
    tokens1 = tokenize(query1, 3)
    tokens2 = [tokenize(t, 3) for t in query2s]

    tokens1 = ner_weight(tokens1)

    max_sim = -10
    _g = query2s[0]
    best_index = 0
    for i, words2 in enumerate(tokens2):
        weighted2 = ner_weight(words2)
        sim = computeSentenceSimWithWeights(tokens1, weighted2)
        if sim > max_sim:
            max_sim = sim
            _g = words2
            best_index = i
    return float(max_sim), _g, best_index

# def ff_embedding_local(word):
#     return model[word]

if __name__ == '__main__':
    # query = ''
    # while query != 'exit':
    #     query = input('>> ')
    #     print(ff_embedding_local(query.strip()))

    q1='我能,用,支付宝,付款,吗'
    q2=["xxxxx"]
    print(ff_embedding('hello'))