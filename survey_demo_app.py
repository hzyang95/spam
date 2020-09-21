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
from module.summary.extract_sentence import Test

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
socketio = SocketIO(generate, async_mode=async_mode)
thread = None
thread_lock = Lock()

webcrawler = WebCrawler()
tester = Test()


class SearchForm(Form):
    choice = SelectField('',
                         choices=[('0', '---选择样例---'), ('1', '北京出现新疫情'), ('2', '新型冠状病毒'), ('3', '人工智能'), ('4', '自动驾驶')],
                         default='0')
    submit = SubmitField('生成')
    post0 = StringField('', render_kw={"placeholder": "输入题目"})
    # postdate = StringField('日期', render_kw={"placeholder": "日期"})
    post1 = StringField('一级标题1', render_kw={"placeholder": "输入标题"})
    post2 = StringField('一级标题2', render_kw={"placeholder": "输入标题"})
    post3 = StringField('一级标题3', render_kw={"placeholder": "输入标题"})
    post4 = StringField('一级标题4', render_kw={"placeholder": "输入标题"})
    post5 = StringField('一级标题5', render_kw={"placeholder": "输入标题"})
    # submit1 = SubmitField('清空')


class FactForm(Form):
    submit = SubmitField('分析')
    choice = SelectField('', choices=[('0', '---选择样例---'),
                                      ('1', '三文鱼感染新冠病毒并传播给人的可能性几乎为0')],
                         default='0')
    boolean = BooleanField(label='篇章级别', default=False)
    post0 = StringField('', render_kw={"placeholder": "输入待分析内容"})
    # postdate = StringField('日期', render_kw={"placeholder": "日期"})
    # submit1 = SubmitField('清空')


@generate.route('/', methods=['GET', 'POST'])
def home():
    post0 = None
    results = [None, None, None]
    er = 0
    searchForm = SearchForm()
    raw = ""
    text = ['---请选择样例填充---', '北京出现新疫情', '新型冠状病毒调研', '人工智能', '自动驾驶']
    # content_word_list = ['价值', '应用', '方法', '问题', '发展']

    content_word_list = [[],
                         ['新疫情源自哪里', '北京疫情现状', '采取的措施', '', ''],
                         ['新型冠状病毒的原理', '新型冠状病毒的传播途径', '新型冠状病毒的预防方案', '新型冠状病毒的疫苗研究', '新型冠状病毒的死亡率'],
                         ['什么是人工智能', '技术成就智能', '人工智能应用之道', '诸多问题亟待解决', '人工智能前景光明'],
                         ['传统出行方式的弊端', '行业宏观形势', '关键技术探究', '急需克服的难题', '行业前景光明']]
    # ['无人机基本概念','广泛的应用场景','国内外研究现状','关键技术的探究','乐观的发展前景']

    # if searchForm.submit.data and searchForm.validate_on_submit():
    if request.method == 'POST':
        # post0 = searchForm.post0.data
        # post1 = searchForm.post1.data
        # post2 = searchForm.post2.data
        # post3 = searchForm.post3.data
        # post4 = searchForm.post4.data
        # post5 = searchForm.post5.data
        li = request.form().count
        print('---', li)
        post0 = request.form['post0']
        post1 = request.form['post1']
        post2 = request.form['post2']
        post3 = request.form['post3']
        post4 = request.form['post4']
        post5 = request.form['post5']
        # postdate = searchForm.postdate.data
        postdate = '2021-01-01'
        key = [i.strip() for i in [post1, post2, post3, post4, post5] if i.strip() != '']
        logger.info(str(post0))
        logger.info(str(key))
        if post0 == '':
            er = 1
        else:
            if len(key) == 0:
                key = [post0]
            time1 = time.time()
            results, raw, URLSS = get_survey(post0, key, postdate, 5, webcrawler)
            # results, raw = write_article(post0, ' '.join(key).strip())
            time2 = time.time()
            logger.info("raw: {}".format(raw.replace('\n', '|')))
            logger.info("time cost: {}s".format(time2 - time1))

    return render_template("home.html", example_inputs_text=content_word_list, example_inputs=text, error=er,
                           form=searchForm, post=post0, results_cont=results, raw=raw)


@generate.route('/fact', methods=['GET', 'POST'])
def fact():
    post0 = None
    results = [None, None, None]
    er = 0
    searchForm = FactForm()
    raw = ""
    urls = []
    text = ['---请选择样例填充---',
            '三文鱼感染新冠病毒并传播给人的可能性几乎为0']
    # 新型冠状病毒可以通过空气传播
    # 中国疫情二次爆发是大概率事件？
    # 抗生素能有效预防和治疗新型冠状病毒
    # 核酸检测结果阴性不能排除新型冠状病毒感染
    nc = None
    passa = None
    # '英超官方确认暂停比赛'

    # '勤洗手和戴口罩可以降低新型冠状病毒感染风险',
    if searchForm.submit.data and searchForm.validate_on_submit():
        post0 = searchForm.post0.data
        boolean = searchForm.boolean.data
        # print(boolean)
        if boolean is True:
            passa = boolean
        # postdate = searchForm.postdate.data
        postdate = '2021-01-01'
        key = []
        logger.info(str(post0))
        logger.info(str(key))
        if post0 == '':
            er = 1
        else:
            if len(key) == 0:
                key = [post0]
            time1 = time.time()
            if boolean:
                results, raw, URLSS = get_survey(post0, key, postdate, 10, webcrawler)
                results, urls = stance_pas(results)
            else:
                results, raw, urls = get_stance_survey(post0, key, postdate, 5, webcrawler)
                # results, urls = stance_sent(results)
                # print(len(results[0][1]))
                # print(len(results[0][2]))
                if len(results[0][1]) == 0 and len(results[0][2]) == 0:
                    nc = 1
            # print(results)
            # print(urls)
            # results, raw = write_article(post0, ' '.join(key).strip())
            time2 = time.time()
            logger.info("raw: {}".format(raw.replace('\n', '|')))
            logger.info("time cost: {}s".format(time2 - time1))

    # return render_template("home.html", example_inputs_text=content_word_list, example_inputs=text, error=er,
    #                        form=searchForm, post=post0, results_bg=results[0], results_ed=results[2],
    #                        results_cont=results[1], raw=raw)
    print(passa)
    return render_template("fact_.html", example_inputs=text, error=er,
                           form=searchForm, post=post0, results_cont=results, raw=raw, urls=urls, nc=nc, passa=passa)
    # return render_template('home.html')


all_token = 0
done_token = 0
pro_now = 0


def background_thread():
    global all_token, done_token, pro_now
    no = -1
    st = -1
    while True:
        socketio.sleep(3)

        no = pro_now
        # print(no)
        if st != no:
            print("=====")
            print(no)
            print(st)
            print("=====")
            text = "正在生成(" + str(no - 4) + "~" + str(no) + "/" + str(all_token) + "）"
            st = no
            socketio.emit('server_response',
                          {'data': text},
                          namespace='/test')


@socketio.on('connect', namespace='/test')
def socket_connect():
    global thread
    with thread_lock:
        if thread is None:
            socketio.start_background_task(target=background_thread)


if __name__ == '__main__':
    socketio.run(generate, debug=True)

    # write_article(
    #     '人工智能',
    #     ' '.join(['什么是人工智能'])
    # )
    # print(get_stance_survey('特朗普请巫师到白宫施法以驱除新冠病毒', ['特朗普请巫师到白宫施法以驱除新冠病毒'], '2021-01-01', 10,webcrawler))
    # results, raw, URLSS = get_survey('特朗普请巫师到白宫施法以驱除新冠病毒', ['特朗普请巫师到白宫施法以驱除新冠病毒'], '2021-01-01', sentnum=10)
    # print(stance_sent(results))

# n = '12345'
# for i in range(len(n)):
#     for j in range(i, len(n)):
#         res[j - i][j] += min(n[j], n[j - i]) * res[j - i + 1][j - 1]
