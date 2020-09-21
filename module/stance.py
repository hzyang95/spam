import asyncio

import tornado.gen
import tornado.httpclient
import tornado.ioloop
from tornado import gen
import json
from functools import partial

# f = './checkpoints/newmixfulllongbigvoc.pt'
# f = './checkpoints/stance_sentmixaveronlysingleshort/model.pt'
# st = StanceDetector(f)
from module.tools import get_res_from_sever


def get_json_sent(data_test, query, gra):
    sample = []
    for item in data_test:
        # print(st.mk(query, item[0]))
        sample.append({'text': item[0], 'question': query})
        # print(query)
        # print(item[0])
        # print(st.mk(query,item[0]))
    # print(st.mk(query, at))
    return {'gra': gra, 'sample': sample}


def get_json_pas(data_test, query, gra):
    sample = []
    at = ''
    for item in data_test:
        # print(st.mk(query, item[0]))
        # sample.append({'text': item[0], 'question': query})
        # print(query)
        # print(item[0])
        # print(st.mk(query,item[0]))
        at += item[0]
    # print(st.mk(query, at))
    sample.append({'text': at, 'question': query})
    return {'gra': gra, 'sample': sample}

def get_json_pas_(data_test, query, gra):
    sample = []
    at = ''
    for item in data_test:
        # print(st.mk(query, item[0]))
        # sample.append({'text': item[0], 'question': query})
        # print(query)
        # print(item[0])
        # print(st.mk(query,item[0]))
        at += item
    # print(st.mk(query, at))
    sample.append({'text': at, 'question': query})
    return {'gra': gra, 'sample': sample}

# URL = 'http://localhost:56789/'


URL_pas = 'http://39.98.138.178:34567/'
URL = 'http://39.98.138.178:45678/'


def stance_sent(content):
    dicts = []
    for item in content:
        topic = item[0]
        sentence = item[1]
        dicts.append(get_json_sent(sentence, topic, 'sent'))

    res = get_res_from_sever(dicts, URL)

    idxs = [eval(i.body)['res'] for i in res]
    mx = [eval(i.body)['mx'] for i in res]

    assert len(idxs) == len(dicts)
    print(idxs)
    print(mx)
    n_cont = []
    urls = []
    color_dict = {0: 'rgba(232,27,22,0.74)', 1: 'rgb(0,0,0)', 2: 'rgba(0,10,232,0.74)'}
    for i, items in enumerate(idxs):
        cont = content[i]
        topic = cont[0]
        sent = cont[1]
        n_sent_0 = []
        n_sent_1 = []
        assert len(sent) == len(items)
        for j, _sent in enumerate(sent):
            # print(items[j])
            color = color_dict[items[j]]
            # print(color)
            url = _sent[1]
            if url not in urls:
                urls.append(url)
            if items[j] == 0:
                n_sent_0.append(_sent + [color])
            if items[j] == 2:
                n_sent_1.append(_sent + [color])
        print(items[-1])
        n_cont.append([topic, n_sent_0, n_sent_1, items[-1]])
    # res = []
    return n_cont, urls


def stance_pas(content):
    dicts = []
    for item in content:
        topic = item[0]
        sentence = item[1]
        dicts.append(get_json_pas(sentence, topic, 'para'))
    print(dicts)
    res = get_res_from_sever(dicts, URL_pas)

    idxs = [eval(i.body)['res'] for i in res]
    mx = [eval(i.body)['mx'] for i in res]

    assert len(idxs) == len(dicts)
    print(idxs)
    print(mx)
    n_cont = []
    urls = []
    color_dict = {0: 'rgba(232,27,22,0.74)', 1: 'rgb(0,0,0)', 2: 'rgba(0,10,232,0.74)'}
    word_dict = {0: '假', 1: '疑', 2: '真'}
    for i, items in enumerate(idxs):
        cont = content[i]
        topic = cont[0]
        sent = cont[1]
        n_cont.append([topic, sent, [word_dict[items[0]], color_dict[items[0]]]])
        for j, _sent in enumerate(sent):
            url = _sent[1]
            if url not in urls:
                urls.append(url)
    # res = []
    return n_cont, urls


def stance_pas_(content):
    dicts = []
    topic = content[3]
    for item in content[0]:
        dicts.append(get_json_pas_(item, topic, 'para'))
    print(dicts)
    res = get_res_from_sever(dicts, URL_pas)

    idxs = [eval(i.body)['res'] for i in res]
    mx = [eval(i.body)['mx'] for i in res]

    assert len(idxs) == len(dicts)
    print(idxs)
    print(mx)
    n_cont = []
    urls = []

    s1=[]
    s2=[]
    u1=[]
    u2=[]

    for i, items in enumerate(idxs):
        if items[0]==0:
            s1.append(content[0][i])
            u1.append(content[1][i])
        if items[0]==2:
            s2.append(content[0][i])
            u2.append(content[1][i])

    return (s1,u1,content[2],content[3]),(s2,u2,content[2],content[3]),u1,u2
