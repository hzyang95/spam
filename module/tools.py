import asyncio
import json
from functools import partial

import tornado.httpclient
import tornado.ioloop
from tornado import gen


def get_res_from_sever(dicts, url):
    @gen.coroutine
    def main(_dic):
        http_client = tornado.httpclient.AsyncHTTPClient()
        responses = yield [
            http_client.fetch(url, method='POST', request_timeout=60.0, body=json.dumps(_dic[i]))
            for i in range(len(_dic))  # 请求URL
        ]
        return responses

    res = []
    asyncio.set_event_loop(asyncio.new_event_loop())
    ioloop = tornado.ioloop.IOLoop.current()
    # main = partial(main, _dic=dicts)
    # res=ioloop.run_sync(main)

    for dict in dicts:
        if len(dict['sample'])!=0:
            main = partial(main, _dic=[dict])
            rd=ioloop.run_sync(main)[0]
            res.append(eval(rd.body)['res'])
        else:
            res.append([])
    return res