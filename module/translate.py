import re
import socket
import nltk.tokenize as tk
import jieba
from pyltp import SentenceSplitter


def clean(raw):
    # _dict = {'\t': '', '　': ' ', '，': ',', '。': '.',
    #          '！': '!', '、': ',', '（': '(', '）': ')',
    #          '：': ':', '‘': '\'', '’': '\'', '“': '"', '”': '"'}
    raw = raw.replace('　', ' ')
    raw = raw.replace('，', ',')
    raw = raw.replace('。', '.')
    raw = raw.replace('？', '?')
    raw = raw.replace('！', '!')
    raw = raw.replace('、', ',')
    raw = raw.replace('（', '(')
    raw = raw.replace('）', ')')
    raw = raw.replace('：', ':')
    raw = raw.replace('‘', '\'')
    raw = raw.replace('’', '\'')
    raw = raw.replace('“', '"')
    raw = raw.replace('”', '"')
    raw = raw.replace('"', '\'')
    # raw = raw.replace('|', ',')
    raw = raw.replace('-', ' - ')
    raw = raw.strip()
    return raw


def find_label(s, choice):
    s = s.strip()

    r = r'<(?:[^"\'>]|"[^"]*"|\'[^\']*\')*>|\||((\(|\[|\{|/|\@|\#|\$|\%|\^|\&|\*)+[0-9]*(\)|\]|\}|/|\@|\#|\$|\%|\^|\&|\*)+)'
    # print(s)
    # |\'|\" |\(|\)  |\<|\> |\[|\]
    # |/|\@|\#|\$|\%|\^|\&|\*
    # [\u4e00-\u9fff]+|
    it = re.finditer(r, s)
    results = [match for match in it]
    # print(results)
    i = 0
    bg = 0
    f = []
    while i < len(results):
        now = results[i]
        if now.start() > bg:
            if choice % 2 == 1:
                temp_str = s[bg:now.start()]
                f += find_chinese(temp_str)
            else:
                f.append(s[bg:now.start()])
        # f.append(s[now.start():now.end()])
        f.append((i, s[now.start():now.end()]))
        bg = now.end()
        i += 1
    if len(s) > bg:
        f.append(s[bg:len(s)])
    return f


def find_chinese(s):
    s = s.strip()
    r = r'((\'|\"|\(|\)|\<|\>|\[|\]|\.|\!|/|\@|\#|\$|\%|\^|\&|\*)*[\u4e00-\u9fff]+(\'|\"|\(|\)|\<|\>|\[|\]|\.|\!|/|\@|\#|\$|\%|\^|\&|\*)*)+'
    # print(s)
    # |\'|\" |\(|\)  |\<|\> |\[|\]
    # |/|\@|\#|\$|\%|\^|\&|\*
    # [\u4e00-\u9fff]+|
    it = re.finditer(r, s)
    results = [match for match in it]
    # print(results)
    i = 0
    bg = 0
    f = []
    while i < len(results):
        now = results[i]
        if now.start() > bg:
            f.append(s[now.start():now.end()])
        f.append((i, s[now.start():now.end()]))
        bg = now.end()
        i += 1
    if len(s) > bg:
        f.append(s[bg:len(s)])
    return f


def write_article(raw, choice):
    """
    :param topic: 即调研的关键词
    :return:
    """
    print(choice)
    raw = raw.strip()

    # res = [[item, item] for item in raw.split(' ')]

    r = r'<(?:[^"\'>]|"[^"]*"|\'[^\']*\')*>'

    ports = [0000, 8898, 8899, 8900, 8901, 8902, 8903, 8904, 8905, 8906, 8907, 8908, 8909, 8910, 8911, 8912, 8913, 8914,
             8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 8923]
    # name = [en-zh, zh-en, fr-zh, zh-fr, ar-zh, zh-ar, ru-zh, zh-ru, es-zh, zh-es, de-zh, zh-de, cs-zh, zh-cs, it-zh, zh-it, nl-zh, zh-nl, pt-zh, zh-pt]
    _language = ['english', 'french', 'english', 'russian', 'spanish', 'german', 'czech', 'italian', 'dutch',
                 'portuguese', 'english', 'english', 'english']

    _raw = raw.split('\n')

    result = []

    for raw in _raw:
        raw = clean(raw)
        # tokenize = tk.sent_tokenize
        if choice % 2 == 0:
            tokenize = SentenceSplitter.split
            sentences = list(tokenize(raw))
        else:
            tokenize = tk.sent_tokenize
            print(_language[int(choice / 2)])
            sentences = tokenize(raw, language=_language[int(choice / 2)])
        # print(sentences)
        res = []
        for sent in sentences:
            # if 说明 % 2 == 1 and check_contain_chinese(sent):
            #     sent = '\n'.join(write_article(sent, 说明 + 1))
            sent = find_label(sent, choice=choice)
            print(sent)
            t_r = []
            for s in sent:
                if type(s) is not str:
                    t_r.append(s[1])
                    continue
                # s = s.strip()
                if not s:
                    continue
                # print(''.join([i for i in list(s) if i in ' ,.<>?/-_+=|[]{}()\\\'\";:*&^%$#@!~`\n\t']))
                if s in ['"', '\'']:
                    continue

                if ''.join([i for i in list(s) if i in ' ,.<>?/-_+=|[]{}()\\\'\";:*&^%$#@!~`\n\t']) == s:
                    t_r.append(s)
                    continue

                # if s[0] in ['"', '\''] and s[-1] not in ['"', '\'']:
                #     s = s[1:]
                # if s[-1] in ['"', '\''] and s[0] not in ['"', '\'']:
                #     s = s[:-1]

                if int(choice) == 13:
                    if len(s) > 0:
                        if s[-1] != '.':
                            s += '.'
                print("raw: " + str(s))
                # t_r.append(s)

                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect(('127.0.0.1', ports[choice]))
                client.sendall(s.encode())
                client.send('\n\n'.encode())
                raw_outputs = None
                while True:
                    data = client.recv(1024).decode()
                    if raw_outputs is None:
                        raw_outputs = data
                    else:
                        raw_outputs += data
                    if raw_outputs.endswith('\n\n'):
                        break
                if choice == 2 and 'a' < raw_outputs[0] < 'z':
                    raw_outputs = list(raw_outputs)
                    raw_outputs[0] = chr(ord(raw_outputs[0]) - 32)
                    raw_outputs = ''.join(raw_outputs)
                temp_res = raw_outputs[:-2].strip()
                if int(choice) == 13:
                    if len(temp_res) > 7:
                        # print(temp_res[-7:])
                        print(temp_res[-7:] == '[zh_CN]')
                        if temp_res[-7:] == '[zh_CN]':
                            # print(temp_res[-7:])
                            temp_res = temp_res[:-7]
                            # print(temp_res)
                print("res: " + str(temp_res))
                if choice % 2 != 0:  # not in [2, 4, 6, 8, 10, 12, 14]
                    temp_res = temp_res.replace(' ', '')
                t_r.append(temp_res)
                client.close()
            res.append(''.join(t_r))
        res = ' '.join(res)
        result.append(res)
    result = ' '.join(result)
    return result
