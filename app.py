# -*- coding: utf-8 -*-
import sys
import time
import logging
from threading import Lock

from flask import Flask, render_template, request
# from flask.ext.wtf import Form
# from flask.ext.bootstrap import Bootstrap
from flask_wtf import Form
from flask_bootstrap import Bootstrap
from wtforms import StringField, SubmitField, SelectField, BooleanField
from flask_socketio import SocketIO

from module.crawler import WebCrawler
from module.write_article import get_stance_survey, get_survey
from module.stance import stance_pas

t = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))
file = logging.FileHandler('log/sglog' + t, encoding='utf-8')
file.setLevel(level=logging.INFO)
logger.addHandler(file)
logger.setLevel(logging.INFO)

logger.info('begin')

async_mode = None
generate = Flask(__name__)
generate.config['SECRET_KEY'] = '961018961018'
bootstrap = Bootstrap(generate)
# socketio = SocketIO(generate, async_mode=async_mode)
thread = None
thread_lock = Lock()

webcrawler = WebCrawler()


class FactForm(Form):
    submit = SubmitField('分析')
    choice = SelectField('', choices=[('0', '---选择样例---'),
                                      ('1', '三文鱼感染新冠病毒并传播给人的可能性几乎为0')],
                         default='0')
    boolean = BooleanField(label='篇章级别', default=False)
    boolean1 = BooleanField(label='篇章级别', default=False)
    post0 = StringField('', render_kw={"placeholder": "输入待分析内容"})
    # postdate = StringField('日期', render_kw={"placeholder": "日期"})
    # submit1 = SubmitField('清空')


@generate.route('/', methods=['GET', 'POST'])
def home():
    post0 = None
    post1 = None
    post2 = None
    post3 = None
    post4 = None
    post5 = None
    results = [None, None, None]
    er = 0
    searchForm = FactForm()
    raw = ""
    URLSS = []
    # text = ['---请选择样例填充---',
    #         '三文鱼感染新冠病毒并传播给人的可能性几乎为0']
    # 新型冠状病毒可以通过空气传播
    # 中国疫情二次爆发是大概率事件？
    # 抗生素能有效预防和治疗新型冠状病毒
    # 核酸检测结果阴性不能排除新型冠状病毒感染
    text = ['---请选择样例填充---',
            '西瓜为什么不能挖着吃？知道真相的我无言以对',
            '鸽子为什么咕咕叫？多数人都不知道',
            '北京出现新疫情'
            # '新型冠状病毒调研'
            ]

    content_word_list = [[],
                         # ['不卫生','胃口被撑大','肚子变圆','',''],
                         ['', '', '', '', ''],
                         ['', '', '', '', ''],
                         ['新疫情源自哪里', '北京疫情现状', '采取的措施', '', ''],
                         # ['新型冠状病毒的原理', '新型冠状病毒的传播途径', '新型冠状病毒的预防方案', '新型冠状病毒的疫苗研究', '新型冠状病毒的死亡率'],
                         ]
    bjh = 1
    qeh = 1
    tth = 1
    oth = 0

    _num = 7

    # '英超官方确认暂停比赛'

    # '勤洗手和戴口罩可以降低新型冠状病毒感染风险',
    if request.method == 'POST':

        print(request.form)
        post0 = request.form['post0']
        post1 = request.form['post1']
        post2 = request.form['post2']
        post3 = request.form['post3']
        post4 = request.form['post4']
        post5 = request.form['post5']

        try:
            bjh = int(request.form['bjh'])
        except:
            bjh = 0
        try:
            qeh = int(request.form['qeh'])
        except:
            qeh = 0
        try:
            tth = int(request.form['tth'])
        except:
            tth = 0
        try:
            oth = int(request.form['oth'])
        except:
            oth = 0

        if oth == 1:
            tth = 0
            bjh = 0
            qeh = 0
        if not (tth or bjh or qeh):
            oth = 1

        _num = int(request.form['_num'])

        print(_num)
        print(bjh, tth, qeh, oth)
        key = []
        postdate = '2021-01-01'
        key = [i.strip() for i in [post1, post2, post3, post4, post5] if i.strip() != '']
        
        tm = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

        logger.info('time: '+tm)

        logger.info('title: '+str(post0))
        logger.info(str(key))

        sites = []
        if oth == 0:
            if bjh == 1:
                sites.append('baijiahao.baidu.com')
            if qeh == 1:
                sites.append('om.qq.com')
            if tth == 1:
                sites.append('sohu.com')

        if post0 == '':
            er = 1
        else:
            if len(key) == 0:
                key = [post0]
            time1 = time.time()
            results, raw, URLSS = get_survey(post0, key, postdate, _num, webcrawler, sites)
            # results, raw = write_article(post0, ' '.join(key).strip())
            time2 = time.time()
            logger.info("raw: {}".format(raw.replace('\n', '|')))
            logger.info("time cost: {}s".format(time2 - time1))

    return render_template("fact.html", example_inputs_text=content_word_list,
                           example_inputs=text, error=er,
                           post0=post0, post1=post1, post2=post2, post3=post3, post4=post4, post5=post5,
                           results_cont=results, raw=raw, urls=URLSS,
                           _num=_num,
                           _tth=tth, _qeh=qeh, _bjh=bjh, _oth=oth)
    # return render_template('home.html')


if __name__ == '__main__':
    generate.run(debug=True)
