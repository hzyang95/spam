# -*- coding: utf-8 -*-
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from urllib import request
import urllib
import ssl
import json
import re
from bs4 import BeautifulSoup as BS
import requests
import logging

logging.getLogger("requests").setLevel(logging.WARNING)
import time
from goose3 import Goose
from module.html_extractor import CxExtractor

ssl._create_default_https_context = ssl._create_unverified_context
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, compress',
    'Accept-Language': 'en-us;q=0.5,en;q=0.3',
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1 (KHTML, like Gecko) CriOS/78.0.3904.108 Mobile/13B143 Safari/601.1.46'
}  # 定义头文件，伪装成浏览器
# headers = {
#     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#     'Accept-Encoding': 'gzip, deflate, compress',
#     'Accept-Language': 'en-us;q=0.5,en;q=0.3',
#     'Cache-Control': 'max-age=0',
#     'Connection': 'keep-alive',
#     'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
# }  # 定义头文件，伪装成浏览器
baike_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
WEB_RESULT_NUMBER = 10

import random

import re

import hashlib


class WebCrawler:
    def __init__(self, result_number=WEB_RESULT_NUMBER, mode='multi'):
        self.result_number = result_number  # 默认保留五条检索数据
        self.mode = mode

    def _get_page_results(self, query, page_idx=0, sites=None):
        results = []
        q_query = urllib.parse.quote(query)
        if sites is not None:
            path = 'http://www.baidu.com.cn/s?wd=' + q_query +'%20site:'+ sites +'&pn=' \
                   + str(page_idx) + '&oq=' + q_query + '&tn=06074089_11_dg&rsv_spt=3'  # word为关键词，pn是百度用来分页的..
        else:
            path = 'http://www.baidu.com.cn/s?wd=' + q_query + '&pn=' \
                + str(page_idx) + '&oq=' + q_query + '&tn=06074089_11_dg&rsv_spt=3'  # word为关键词，pn是百度用来分页的..
        # path = 'http://www.so.com/s?q=' + urllib.parse.quote(query)  # word为关键词，pn是百度用来分页的..
        # https://www.baidu.com/s?wd= &pn= &oq= &tn=06074089_11_dg
        # print(path)
        # path = 'http://www.baidu.com.cn/s?wd=' + urllib.parse.quote(query) + '&pn=' + str(page_idx)  # word为关键词，pn是百度用来分页的..
        # proxy_ip = {
        #     'http': random.choice(http_ip),
        # }
        print(path)
        response = requests.get(url=path, headers=headers)
        # print(response)
        soup = BS(response.content, 'html.parser')
        contents = soup.find_all(name='div', attrs={'class': 'c-container'})
        urls = []

        def proc(href):
            baidu_url = requests.get(url=href, headers=headers, allow_redirects=False)
            real_url = baidu_url.headers['Location']  # 得到网页原始地址
            return real_url

        for content in contents:
            try:
                title = content.a.text
                abstract = content.find_all(name='div', attrs={'class': 'c-abstract'})
                if abstract:
                    abstract = abstract[0].text
                elif content.find_all(name='div', attrs={'class': 'c-span-last'})[0].text:
                    abstract = content.find_all(name='div', attrs={'class': 'c-span-last'})[0].text
                else:
                    continue
                href = content.a.get('href')
                if href[:4] == 'http':
                    urls.append(href)

                # baidu_url = requests.get(url=href, headers=headers, allow_redirects=False)
                # real_url = baidu_url.headers['Location']  # 得到网页原始地址
                # results.append((real_url, title, abstract))
                if self.mode == 'multi':
                    results.append([href, title, abstract])
                else:
                    baidu_url = requests.get(url=href, headers=headers, allow_redirects=False)
                    real_url = baidu_url.headers['Location']  # 得到网页原始地址
                    results.append((real_url, title, abstract))
            except:
                continue
        if self.mode == 'multi':
            try:
                with ThreadPool(len(urls)) as threads:
                    result = threads.map(proc, urls)
                for i in range(len(result)):
                    results[i] = (result[i], results[i][1], results[i][2])
            except:
                pass
        return results

    def _filter_results(self, results):
        filtered_results = []
        for url, title, abstract in results:
            # if url.endswith('.com/') or url.endswith('.net/') or url.endswith(
            #         '.cn/') or 'zhihu' in url or 'tieba' in url or 'wenku.baidu.com' in url or \
            #         'tieba.baidu.com' in url :
            #     # print(url)
            #     continue
            # 'vp.fact.qq.com' in url or
            if 'tieba' in url or 'wenku.baidu.com' in url or \
                    'tieba.baidu.com' in url:
                # print(url)
                continue
            if '...' in abstract:
                abstract = abstract[:abstract.find('...') + 3]
            filtered_results.append(url)
        # random.shuffle(filtered_results)
        return filtered_results

    def get_web_results(self, query, postdate,nums, sites=None, ):

        def if_url2io_available():
            url = 'http://www.360doc.com/content/20/0201/21/15607564_889110503.shtml'
            r = requests.get(
                'http://url2api.applinzi.com/article?token=9lrJ7uvLTzKxRf-r7mKOyA&url={}&fields=text'.format(
                    urllib.parse.quote(url)))
            if r.status_code == 603:
                return False
            else:
                return True

        stat = if_url2io_available()
        print(stat)
        page_idx = 0
        results = []
        urls = []
        # print(postdate)
        while len(results) < nums:
            time1 = time.time()
            page_results = self._filter_results(self._get_page_results(query, page_idx,sites))
            time2 = time.time()

            page_idx += 10
            if page_idx > 50:
                break

            # print('2 '+str(time2 - time1))
            def proc_url2io(url):

                # m.update(url.encode('utf-8'))
                # hash_value = ''.join([i for i in list(url) if 'a' <= i <= 'z' or 'A' <= i <= 'Z' or '0' <= i <= '9' or i=='_'])
                m=hashlib.md5()
                m.update(url.encode('utf-8'))
                hash_value = m.hexdigest()

                if os.path.exists('web_data/html/' + hash_value):
                    print('get ' + url + ' ' + hash_value)
                    with open('web_data/html/' + hash_value, 'r', encoding='utf-8') as f:
                        r = f.read().strip()
                else:
                    print('search ' + url)
                    ret = requests.get(
                        'http://url2api.applinzi.com/article?token=9lrJ7uvLTzKxRf-r7mKOyA&url={}&fields=text'.format(
                            urllib.parse.quote(url)))
                    with open('web_data/html/' + hash_value, 'w', encoding='utf-8') as f:
                        try:
                            print('save' + str(hash_value))
                            r = ret.json()['text']
                            # print(ret)
                        except:
                            r = ''
                        f.write(r)
                return r

            def ser_date(_str):
                pt1 = re.compile('(\d{4}([-年])\d{1,2}([-月])\d{1,2})').search(_str)
                pt2 = re.compile('(\d{1,2}([-月])\d{1,2})').search(_str)
                pt3 = re.compile('(\d{4}([-年])\d{1,2})').search(_str)
                _date = None
                if pt1 is not None:
                    _date = pt1.group()
                elif pt3 is not None:
                    _date = pt3.group()
                elif pt2 is not None:
                    _date = pt2.group()
                _date = _date.replace('年', '-')
                _date = _date.replace('月', '-')
                # _date = _date.replace('/', '-')
                temp_list = _date.split('-')
                # print(temp_list)
                if len(temp_list[1]) == 1:
                    temp_list[1] = '0' + temp_list[1]
                if len(temp_list) == 3:
                    if int(temp_list[0]) > 2020 or int(temp_list[0]) < 2000:
                        _date = ''
                    if int(temp_list[1]) > 12 or int(temp_list[1]) < 1:
                        _date = ''
                    if int(temp_list[2]) > 31 or int(temp_list[2]) < 1:
                        _date = ''
                    if len(temp_list[2]) == 1:
                        temp_list[2] = '0' + temp_list[2]
                else:
                    if len(temp_list[0]) == 4:
                        if int(temp_list[0]) > 2020 or int(temp_list[0]) < 2000:
                            _date = ''
                        if int(temp_list[1]) > 12 or int(temp_list[1]) < 1:
                            _date = ''
                    else:
                        if len(temp_list[0]) == 1:
                            temp_list[0] = '0' + temp_list[0]
                        if int(temp_list[0]) > 12 or int(temp_list[0]) < 1:
                            _date = ''
                        if int(temp_list[1]) > 31 or int(temp_list[1]) < 1:
                            _date = ''
                if _date != '':
                    _date = '-'.join(temp_list)
                # print(_date)
                return _date

            def proc(url, postdate):
                m=hashlib.md5()
                m.update(url.encode('utf-8'))
                hash_value = m.hexdigest()
                # hash_value = ''.join([i for i in list(url) if 'a' <= i <= 'z' or 'A' <= i <= 'Z'])
                if os.path.exists('web_data/html/' + hash_value):
                    with open('web_data/html/' + hash_value, 'r', encoding='utf-8') as f:
                        ret = f.read().strip()
                else:
                    cx = CxExtractor()
                    try:
                        # print("convert " + url)
                        # test_html = cx.getHtml(url)
                        test_html = requests.get(url=url, headers=headers, timeout=(1, 1)).content
                        _str = str(test_html, encoding='utf-8')
                        date = ser_date(_str)
                        if len(date) == 10:
                            if (date > postdate):
                                return ''
                        elif len(date) == 7:
                            if (date > postdate[:7]):
                                return ''
                        else:
                            if (date > postdate[-5:]):
                                return ''
                        # print(date)
                        content = cx.filter_tags(_str)
                        # print(content)
                        ret = cx.getText(str(content))
                        # print("end ")
                    except:
                        # print('error: ' + str(url))
                        ret = ''
                    # with open('web_data/html/' + hash_value, 'w', encoding='utf-8') as f:
                    #     f.write(ret)
                return ret

            print(page_results)
            if len(page_results)==0:
                continue
            if self.mode == 'multi':
                if stat:
                    with ThreadPool(len(page_results)) as threads:
                        crawl_results = threads.map(proc_url2io, page_results)
                else:
                    with ThreadPool(len(page_results)) as threads:
                        crawl_results = threads.map(partial(proc, postdate=postdate), page_results)
            time3 = time.time()
            for i, url in enumerate(page_results):
                if url in urls:
                    continue
                # 利用url2io获取正文数据
                # print(url)
                if self.mode == 'multi':
                    r = crawl_results[i]
                    # print(url)
                    # print(r)
                else:
                    # r = requests.get(
                    #     'http://url2api.applinzi.com/article?token=9lrJ7uvLTzKxRf-r7mKOyA&url={}&fields=text'.format(
                    #         urllib.parse.quote(url)))
                    cx = CxExtractor()
                    try:
                        # print("convert " + url)
                        # test_html = cx.getHtml(url)
                        test_html = requests.get(url=url, headers=headers).content
                        _str = str(test_html, encoding='utf-8')

                        date = ser_date(_str)

                        if len(date) == 10:
                            if (date > postdate):
                                continue
                        elif len(date) == 7:
                            if (date > postdate[:7]):
                                continue
                        else:
                            if (date > postdate[-5:]):
                                continue
                        print(date)
                        content = cx.filter_tags(_str)
                        # print(content)
                        ret = cx.getText(str(content))
                        # print("end " + ret)
                    except:
                        # print('error: ' + str(url))
                        ret = ''
                    r = ret
                if stat:
                    if len(r) == 0:  # 有些页面解析不到text字段，过滤掉
                        continue
                    results.append((url, r))
                else:
                    if len(r) == 0 or r == 'This page has no content to extract':  # 有些页面解析不到text字段，过滤掉
                        continue
                    results.append((url, r))
                urls.append(url)
            # print(len(results))
            time4 = time.time()
            print('4 ' + str(time4 - time2))
        return results[:min(len(results), nums)]
        # return query


if __name__ == "__main__":
    # print('123' in '12534')
    time1 = time.time()
    # # topic = '八角大楼'
    topic = ['北京通报的新冠肺炎病人不含外地人,是真的吗']
    webcrawler = WebCrawler(mode='multi')
    # for i in topic:
    #     web = webcrawler.get_web_results(i)
    with ThreadPool(len(topic)) as threads:
        crawl_results = threads.map(partial(webcrawler.get_web_results, postdate='2020-02-23'), topic)
    print(crawl_results)
    time2 = time.time()
    print(time2 - time1)
