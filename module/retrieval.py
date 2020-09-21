import jieba
import torch
import numpy as np

from scipy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer

from jieba import analyse

from module.tools import get_res_from_sever


def doc_retrieval_tfidf(doc_list, query):
    # print(doc_list)
    text = [' '.join(jieba.cut(' '.join(item))) for item in doc_list]
    ques = [' '.join(jieba.cut(query))]
    tokens = text + ques
    # print(tokens)
    # print(example)
    tfidf = TfidfVectorizer()
    tfidf.fit(tokens)
    x = tfidf.transform(text).toarray()
    y = tfidf.transform(ques).toarray()
    # print(x.shape)
    # print('='*20)
    # print(y.shape)
    output = np.matmul(x, y.transpose())
    # print(res)
    # print(np.argmax(res))
    tops = torch.topk(torch.tensor(output), min(len(text), max(6, len(text) * 2 // 3)), dim=0)
    # print(tops[1])
    return tops[1]


def doc_retrieval_keywords(doc_list, query):
    # print(doc_list)
    query_k = jieba.analyse.extract_tags(query, topK=5, withWeight=True, allowPOS=())
    query_k = {_item[0]: _item[1] for _item in query_k}
    ress = []
    for item in doc_list:
        # keywords = jieba.analyse.extract_tags(' '.join(item), topK=5, withWeight=True, allowPOS=())
        # keywords = jieba.analyse.textrank(item[0], topK=5, withWeight=True, allowPOS=())

        # 访问提取结果
        # keywords = {_item[0]: _item[1] for _item in keywords}
        # print(keywords)
        keywords = list(jieba.cut(' '.join(item)))

        num = 0
        for i in query_k:
            # if i in item[3]:
            if i in keywords:
                # num += query_k[i] * keywords[i]
                num += 1
        ress.append(num)
    tops = torch.topk(torch.tensor(ress), min(3, len(doc_list)), dim=0)
    # print(tops[1])
    return tops[1]


def doc_retri(doc_refore_retri):
    doc_after_retrieval = []
    for item in doc_refore_retri:
        doc_list, urlss, sent, query = item
        # return: [[para1.1,para1.2,...],[para2.1,para2.2,...]

        para_after_retri = []
        url_after_retri = []

        if len(doc_list) > 0:
            index_doc_retrieval = list(doc_retrieval_tfidf(doc_list, query))
            # index_doc_retrieval = list(doc_retrieval_keywords(doc_list, query))

            # print(index_doc_retrieval)
            for i in index_doc_retrieval:
                # print('==============')
                # print(len(doc_list[i]))
                for item in doc_list[i]:
                    # print(len(item))
                    para_after_retri.append(item)
                    url_after_retri.append(urlss[i])
            assert len(para_after_retri) == len(url_after_retri)
        doc_after_retrieval.append((para_after_retri, url_after_retri, sent, query))
    print(doc_after_retrieval)
    return doc_after_retrieval
    # index_para_retrieval = list(select_model.eval(para_to_retri, query))
    # aftr_doc_list = [para_to_retri[i] for i in index_para_retrieval]
    # aftr_urlss = [url_to_retri[i] for i in index_para_retrieval]
    # return aftr_doc_list, aftr_urlss


def get_json(data_test, query, tops, gra):
    sample = []
    for item in data_test:
        sample.append({'text': item, 'question': query})
    return {'gra': gra, 'sample': sample, 'tops': tops, 'question': query}


# URL = 'http://localhost:56789/'
URL = 'http://39.98.138.178:56789/'
URL_sent = 'http://39.98.138.178:56788/'


# URL_sent = 'http://39.98.138.178:56787/'


def sem_retri(dataset, gra):
    dicts = []
    URLSS = []
    print(dataset)
    for item in dataset:
        para_to_retri, url_to_retri, tops, query = item
        dicts.append(get_json(para_to_retri, query, tops, gra))

    # print(dicts)
    if gra == 'sent':
        URLs = URL_sent
    else:
        URLs = URL

    res = get_res_from_sever(dicts, URLs)

    idxs = res

    assert len(idxs) == len(dataset)
    print(idxs)
    res = []

    for i, item in enumerate(dataset):
        para_to_retri, url_to_retri, tops, query = item
        temp = 0
        _now = 0
        if len(para_to_retri) == 0:
            res.append([para_to_retri, url_to_retri, tops, query])
            continue
        tfidf = TfidfVectorizer()
        tfidf.fit([' '.join(jieba.cut(_text)) for _text in para_to_retri])
        _temp_index = []

        while temp < tops:
            if _now >= len(idxs[i]):
                break
            content = para_to_retri[idxs[i][_now]]
            if len(content) < 15:
                _now += 1
                continue
            textB = [' '.join(jieba.cut(content))]
            y = tfidf.transform(textB).toarray()
            flag = 0
            for index in range(len(_temp_index)):
                _temp_content = para_to_retri[_temp_index[index]]
                textA = [' '.join(jieba.cut(_temp_content))]
                try:
                    x = tfidf.transform(textA).toarray()
                    _norm = linalg.norm(x) * linalg.norm(y)
                    if _norm != 0:
                        output = np.matmul(x, y.transpose()) / (linalg.norm(x) * linalg.norm(y))
                    else:
                        output = 0
                    if output > 0.5:
                        # print(str(output)+' '+_temp_title+' '+content)
                        flag = 1
                        break
                except:
                    pass
            if flag == 0:
                temp += 1
                _temp_index.append(idxs[i][_now])
            _now += 1
        _temp_index.sort()
        aftr_doc_list = [para_to_retri[_i] for _i in _temp_index]
        aftr_urlss = [url_to_retri[_i] for _i in _temp_index]
        # aftr_doc_list = [para_to_retri[_i] for _i in idxs[i]]
        # aftr_urlss = [url_to_retri[_i] for _i in idxs[i]]
        URLSS += aftr_urlss
        res.append([aftr_doc_list, aftr_urlss, tops, query])
        # print(aftr_doc_list)

    return res, URLSS
