import codecs
import json
import os
import random
from functools import partial

from multiprocessing.pool import ThreadPool

from pyltp import SentenceSplitter

from module.retrieval import sem_retri, doc_retri

# paragraph retrieval + summary
from module.stance import stance_pas_
from module.translate import write_article


def get_survey(topic, topic_sent, postdate, sentnum, webcrawler, sites=[]):
    """
    :param sentnum:
    :param postdate:
    :param topic_sent:
    :param topic: 即调研的关键词
    :return:
    """
    global all_token, done_token

    # 预处理用户输入
    topic_to_search = get_topic_to_search(topic, topic_sent, sentnum)

    # paragraph retrieval + summary
    # results = para_summary(topic_to_search, webcrawler, tester, postdate)

    # paragraph retrieval + sentence retrieval
    results, URLSS = para_sent(topic_to_search, webcrawler, postdate, sites)

    # 结果后处理
    article, raw = post_process(results, topic, topic_sent)

    return article, raw, URLSS
    # return results


def get_stance_survey(topic, topic_sent, postdate, sentnum, webcrawler):
    """
    :param webcrawler:
    :param sentnum:
    :param postdate:
    :param topic_sent:
    :param topic: 即调研的关键词
    :return:
    """
    global all_token, done_token

    color_dict = {0: 'rgba(232,27,22,0.74)', 1: 'rgba(0,10,232,0.74)'}
    ti = {0: '假', 1: '真'}

    # 预处理用户输入
    topic_to_search = get_topic_to_search(topic, topic_sent, sentnum)

    doc_after_crawl = get_doc(topic_to_search, webcrawler, postdate)
    print(doc_after_crawl)
    doc_after_stace = stance_pas_(doc_after_crawl[0])
    print(doc_after_stace)

    raw = topic_to_search[0][0] + '\n'

    _res = [[], []]
    URLSS = []

    _list_Sent = []
    for _, st in enumerate(doc_after_stace[:2]):
        if len(st[0]) == 0:
            continue
        raw += ti[_]
        doc_aft_retri = doc_retri([st])
        print('doc complete')

        paras_aft_retri, unused = sem_retri(doc_aft_retri, 'para')
        print("para complete")

        sent_before_retri = sentence_split(paras_aft_retri, topic_to_search)
        sent_aft_retri, URLSS_ = sem_retri(sent_before_retri, 'sent')

        URLSS += URLSS_
        results = []

        for i in sent_aft_retri:
            # print(i)
            for ind, item in enumerate(i[0]):
                if item not in _list_Sent:
                    _list_Sent.append(item)
                    if item[:3] == '因此，' or item[:3] == '所以，':
                        item = item[3:]
                    raw += item
                    # print(i[0])
                    # print(i[1])
                    # print(_)
                    # print(ind)
                    results.append([item, i[1][ind], color_dict[_]])
        _res[_] = results

    # 结果后处理
    # article, raw = post_process(results, topic, topic_sent)
    URLSS = list(set(URLSS))
    # URLSS = doc_after_stace[2:]
    return [[topic_to_search[0][0], _res[0], _res[1]]], raw, URLSS
    # return results


def para_summary(topic_to_search, webcrawler, tester, postdate):
    """
    :param topic_to_search:
    :param webcrawler:
    :param tester:
    :param postdate:
    :return:
    """

    def proc(_item):
        query, sent = _item
        URLSS = []
        web_data_path = os.path.join('./web_data', query + '_summary.json')
        if not os.path.exists(web_data_path):
            web_result = webcrawler.get_web_results(query, postdate)
            # print(web_result)
            para_to_retri, url_to_retri = format_doc(web_result)  # 将搜索结果格式化为抽取模型的输入形式，并保存到文件
            para_after_retri, url_after_retri, sent, query = doc_retri([[para_to_retri, url_to_retri, sent, query]])[0]
            index_para_retrieval, URLSS = sem_retri([[para_after_retri, url_after_retri, sent, query]], 'para')
            print(index_para_retrieval)
            save_to_file(index_para_retrieval, True)
        output = tester.test(sent, web_data_path)[0]
        return output, URLSS

    with ThreadPool(len(topic_to_search)) as threads:
        results = threads.map(proc, topic_to_search)

    results = [i[0] for i in results]
    URLS = [i[1] for i in results]
    URLS = list(set(URLS))

    return results, URLS


# paragraph retrieval + sentence retrieval
def para_sent(topic_to_search, webcrawler, postdate, sites=[]):
    """
    :param sites:
    :param topic_to_search:
    :param webcrawler:
    :param postdate:
    :return:
    """
    key_words_to_crawl = get_keywords_to_retri(topic_to_search)

    URLSS = []
    print('begin')
    print(key_words_to_crawl)

    if len(key_words_to_crawl) != 0:
        doc_before_retri = get_doc(key_words_to_crawl, webcrawler, postdate, sites)
        print("crawl complete")
        print(doc_before_retri)

        # if len(doc_before_retri[0][0]) == 0:
        #     return [[]], []

        doc_aft_retri = doc_retri(doc_before_retri)
        print('doc complete')
        # print(doc_aft_retri)

        # paras_aft_retri, _ = sem_retri(doc_aft_retri, 'para')
        # print("para complete")
        # print(paras_aft_retri)

        sent_before_retri = sentence_split(doc_aft_retri, key_words_to_crawl)
        sent_aft_retri, _ = sem_retri(sent_before_retri, 'sent')
        print(sent_aft_retri)
        print("retri complete")

        jss = save_to_file(sent_aft_retri, False)

    results = []
    # for _item in topic_to_search:
    #     query, sent = _item
    #     web_data_path = os.path.join('./web_data', query + '.json')
    #     with codecs.open(web_data_path, 'r', encoding='utf-8') as f:
    #         examples = json.load(f)
    for examples in jss:
        res = []
        for i in examples:
            res.append([examples[i][0], examples[i][1]])
            URLSS.append(examples[i][1])
        results.append(res)
    # print(results)
    URLSS = list(set(URLSS))
    return results, URLSS


def get_topic_to_search(topic, topic_sent, sentnum):
    if len(topic) > 4:
        if topic[-4:] == '调研报告':
            topic = topic[:-4]
        if topic[-5:] == '的调研报告' and topic[:2] == '关于':
            topic = topic[2:-5]
        if topic[-2:] == '调研':
            topic = topic[:-2]
        if topic[-3:] == '的调研' and topic[:2] == '关于':
            topic = topic[2:-3]
    # topic_sent = topic_sent.split(' ')
    print(topic_sent)

    topic_to_search = []

    for item in topic_sent:
        if topic in item:
            query = item
        else:
            query = topic + ' ' + item
        topic_to_search.append((query, sentnum))

    return topic_to_search


def get_keywords_to_retri(topic_to_search):
    key_words_to_crawl = []
    for _item in topic_to_search:
        query, sent = _item
        # web_data_path = os.path.join('./web_data', query + '.json')
        # if not os.path.exists(web_data_path):
        key_words_to_crawl.append(_item)
    return key_words_to_crawl


def get_doc(key_words_to_crawl, webcrawler, postdate, sites=[]):
    def crawl_and_doc_retri(_item, postdate):
        query, sent = _item
        # if not os.path.exists(web_data_path):
        if len(sites) == 0:
            web_result = webcrawler.get_web_results(query, postdate, 10)
        else:
            web_result = []
            for s in sites:
                web_result += webcrawler.get_web_results(query, postdate, 5, s)
        # print(web_result)
        random.shuffle(web_result)
        mid_res, mid_res_url = format_doc(web_result)

        return mid_res, mid_res_url, sent, query

    with ThreadPool(len(key_words_to_crawl)) as threads:
        doc_list = threads.map(partial(crawl_and_doc_retri, postdate=postdate), key_words_to_crawl)

    return doc_list


def post_process(results, topic, topic_sent):
    print(topic_sent)
    content = []
    for i, item in enumerate(results):
        # aft_trans = []
        # for ii in item:
        #     ori=ii[0]
        #     mid=write_article(ori, 2)
        #     aft=write_article(mid, 1)
        #     aft_trans.append(aft)
        # clean_text = ''.join([ii[0] for ii in item])
        # aft_trans=''.join(aft_trans)
        # print(clean_text)
        clean_text = ''.join([ii[0] for ii in item])
        mid_text = write_article(clean_text, 2)
        aft_trans = write_article(mid_text, 1)
        content.append([topic_sent[i], [clean_text, aft_trans]])
    article = content
    raw = ""
    # print(topic)
    raw += topic + "\n"
    # raw += '\t' + ''.join([ii[0] for ii in introduction]) + '\n'

    for i in content:
        raw += i[0] + '\n'
        raw += '\t' + i[1][1] + '\n'

    return article, raw


def save_to_file(sent_aft_retri, is_summary):
    dl = []
    for i, item in enumerate(sent_aft_retri):
        sents, urls, tops, query = item
        # if is_summary:
        #     web_data_path = os.path.join('./web_data', query + '_summary.json')
        # else:
        #     web_data_path = os.path.join('./web_data', query + '.json')
        assert len(sents) == len(urls)
        dic = {}
        sent_temp = []
        for ind in range(len(sents)):
            sent_temp.append(sents[ind])
            dic[str(ind)] = [sents[ind], urls[ind]]
        # w_f = codecs.open(web_data_path, "w", "utf-8")
        # json.dump(dic, w_f, ensure_ascii=False)
        # w_f.close()
        dl.append(dic)
    return dl


def check_contain_chinese(check_str):
    _all = len(check_str)
    _ch = 0
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff' or '0' <= ch <= '9' or 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
            _ch += 1
    return float(_ch) / float(_all)


def format_doc(web_results):
    """
    对从web上获取的数据进行处理，送到下一步抽取句子
    :param web_results: 利用爬虫爬取的结果
    :return:
    """
    # 因为是多文档，处理为每个文档是一个段落

    doc_list = []  # list of paragraph, every paragraph contains multi sentences
    urlss = []
    for url, text in web_results:
        # print(text)
        sentences = text.split('\n')
        sents = []
        urls = []
        for sent in sentences:
            sent = sent.strip()
            if "引用" in sent or "相关阅读" in sent or "特别声明" in sent or "声明" in sent or "来源" in sent or "原标题" in sent:
                continue
            elif len(sent) < 15:
                continue
            elif check_contain_chinese(sent) < 0.6:
                continue
            else:
                sents.append(sent.strip())
                # print(len(sent.strip()))
                # if len(sent.strip()) < 20:
                #     print(sent)
        doc_list.append(sents)
        urlss.append(url)
    # print('-'*60)
    # print(doc_list)
    # for i in doc_list:
    #     print(len(i))
    # print(len(doc_list))
    # print(len(urlss))

    return doc_list, urlss

    # doc_after_retrieval, urlss = retrieval(query, doc_list, urlss)
    #
    # return doc_after_retrieval, urlss


def sentence_split(paras_aft_retri, topic_to_search):
    assert len(paras_aft_retri) == len(topic_to_search)
    res = []
    for i, item in enumerate(paras_aft_retri):
        para_to_split, urls, tops, query = item
        tops = topic_to_search[i][1]
        sent_aft_split = []
        url_aft = []
        for ind, para in enumerate(para_to_split):
            sents = SentenceSplitter.split(para)
            url = urls[ind]
            for sent in sents:
                if sent[-1] == '？' or sent[-1] == '?':
                    continue
                elif "引用" in sent or "相关阅读" in sent or "特别声明" in sent or "声明" in sent or "来源" in sent \
                        or "原标题" in sent or "联系电话" in sent or "联系方式" in sent:
                    continue
                elif len(sent) < 15:
                    continue
                elif check_contain_chinese(sent) < 0.6:
                    continue
                else:
                    sent_aft_split.append(sent)
                    url_aft.append(url)
        assert len(sent_aft_split) == len(url_aft)
        res.append((sent_aft_split, url_aft, tops, query))
        # print(res)
    return res
