# coding=utf-8
# !/usr/bin/env python3
import json
import models
import module
import argparse, random, logging, numpy, os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
from collections import OrderedDict
import re
import codecs

from pyltp import SentenceSplitter
from pyltp import Segmentor

LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`


segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir', type=str, default='from_xf/')
parser.add_argument('-perl_dir', type=str, default='')
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_num', type=int, default=100)
parser.add_argument('-seg_num', type=int, default=10)
parser.add_argument('-kernel_num', type=int, default=100)
parser.add_argument('-kernel_sizes', type=str, default='3,4,5')
parser.add_argument('-model', type=str, default='LSTM_GRU_t')
parser.add_argument('-hidden_size', type=int, default=96)
parser.add_argument('-elmo_path', default='ELMoForManyLangs/zhs.model')
parser.add_argument('-elmo_dim', type=int, default=1024)
parser.add_argument('-proj_dim', type=int, default=0)
parser.add_argument('-vocab', default='Vocab_ELMo')
parser.add_argument('-bert_path', default='bert/chinese_L-12_H-768_A-12')
parser.add_argument('-bert_batch_size', type=int, default=512)
parser.add_argument('-para_loss_weight', type=float, default=1.0)
parser.add_argument('-use_elmo', action='store_true', default=False)
parser.add_argument('-use_seg', action='store_true')
parser.add_argument('-use_mono', action='store_true')
# train
parser.add_argument('-lr', type=float, default=1e-5)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-epochs', type=int, default=15)
parser.add_argument('-seed', type=int, default=1111)
parser.add_argument('-train_dir', type=str, default='./list/train.json')
parser.add_argument('-val_dir', type=str, default='./list/valid.json')
parser.add_argument('-embedding', type=str, default='./data/vocab.npz')
parser.add_argument('-word2id', type=str, default='./data/vocab.npz')
parser.add_argument('-report_every', type=int, default=10000000)
parser.add_argument('-seq_trunc', type=int, default=100)
parser.add_argument('-max_norm', type=float, default=1.0)
parser.add_argument('-bert_finetuning', action='store_true')
parser.add_argument('-optimize_on_cpu', action='store_true', default=True)
parser.add_argument('-optimi', type=str, default='')
# test
parser.add_argument('-load_dir', type=str, default='./checkpoints/h96_newlr_sub_LSTM_GRU_t_seed1_e8.pt')
parser.add_argument('-test_dir', type=str, default='./test.json')
parser.add_argument('-output_dir', type=str, default='./outputs')
# parser.add_argument('-ref',      type=str,default='/ref')
# parser.add_argument('-hyp',      type=str,default='/hyp')  
# parser.add_argument('-doc',      type=str,default='/doc')
parser.add_argument('-topk', type=float, default=0.5)
parser.add_argument('-bound', type=float, default=-1.0)
parser.add_argument('-predout', action='store_true', default=True)
# device
parser.add_argument('-device', type=int, default=None)
parser.add_argument('-bert_gpu_ids')  # ,default='1,2,3')
# option
parser.add_argument('-test', action='store_false')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-predict', action='store_true')
parser.add_argument('-lostype', type=str, default='')
parser.add_argument('-MLP', type=str, default='')
parser.add_argument('-method', type=str, default='')
parser.add_argument('-lastoutput', action='store_true', default=False)
parser.add_argument('-init', type=str, default='Xavier')
parser.add_argument('-lr_epoch', type=int, default=4)
parser.add_argument('-rl', action='store_true', default=False)
parser.add_argument('-taskID', type=int, default=0)
args = parser.parse_args()
use_gpu = args.device is not None
args.use_gpu = use_gpu
args.bert_gpu_ids = list(map(int, args.bert_gpu_ids.split(','))) if args.bert_gpu_ids else []

# args.ref, args.hyp, args.doc = args.outputs+args.ref, args.outputs+args.hyp, args.outputs+args.doc
# if not os.path.exists(args.ref): os.makedirs(args.ref)
# if not os.path.exists(args.hyp): os.makedirs(args.hyp)  
# if not os.path.exists(args.doc): os.makedirs(args.doc)

# if torch.cuda.is_available() and not use_gpu:
#     print("WARNING: You have a CUDA device, should run with -device 0")

torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logging.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logging.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def make_rouge(ref, hyp):
    if type(ref) != list:
        ref = [ref]
    if type(hyp) != list:
        hyp = [hyp]
    ref_p = os.path.join(args.save_dir, 'ref.tmp')
    hyp_p = os.path.join(args.save_dir, 'hyp.tmp')
    with open(ref_p, 'w', encoding='utf-8') as fr, open(hyp_p, 'w', encoding='utf-8') as fh:
        fr.write('\n'.join(ref))
        fh.write('\n'.join(hyp))
    # a = os.popen('perl ROUGE.pl 1 N ref.tmp hyp.tmp') 
    # b = os.popen('perl ROUGE.pl 2 N ref.tmp hyp.tmp >> rouge.tmp') 
    rouge = os.popen('perl ' + args.perl_dir + ' L N ' + ref_p + ' ' + hyp_p).read().strip()
    # print(rouge.read().strip()) 

    return rouge



class PostEdit(object):
    def __init__(self, summary_results, taskID, sent_len=200):
        results = [(p, s.replace(' ', ''), u) for p, s, u in summary_results]

        self.summary_results = results
        self.taskID = taskID
        self.SENT_LEN = sent_len  # 最终段落的长度，依据此划分段落
        self.output = None
        self._init()

    def _init(self):
        if self.taskID == 0:  # 第一个任务，直接输出
            self.output = [[s, u] for p, s, u in self.summary_results]

        elif self.taskID == 1:  # 第一个任务，直接输出
            self.output = ''.join([s for p, s in self.summary_results])

        elif self.taskID == 4:  # 需要分三段：会议背景概述，会议目标，会议主题，此处写规则，按照1/5，1/5， 3/5
            para_set = [['宣言背景\n'], ['\n宣言目标\n'], ['\n宣言主题\n']]
            output_len = len(''.join([s for p, s in self.summary_results]))
            sent_type = 0
            for _, s in self.summary_results:  # 需要判断每一句的类别
                if sent_type == 2 or len(''.join(para_set[sent_type])) < output_len / 5:
                    sent_type = sent_type
                else:
                    sent_type += 1
                para_set[sent_type].append(s)
            self.output = '\n'.join([''.join(sents) for sents in para_set])

        else:  # 其余的情况，每SENT_LEN长度左右分段
            sent_set = self._split_sent()
            para_set = []  # 段落集合，每个集合里面属于最终结果的一个段落
            for key in sorted(sent_set.keys()):
                sent = ' '.join(sent_set[key])
                if len(para_set) == 0:  # 第一段
                    para_set.append(sent)
                else:
                    if len(para_set[-1]) > self.SENT_LEN:
                        para_set.append(sent)
                    else:
                        para_set[-1] = para_set[-1] + sent
            self.output = '\n'.join(para_set)

    def _split_sent(self):
        sent_set = dict()  # 将得到的句子结果划分为句子集合，每个集合里面的句子属于原文的一个段落
        for position, sent in self.summary_results:
            if position in sent_set:
                sent_set[position].append(sent)
            else:
                sent_set[position] = [sent]
        return sent_set

def get_seg_tag(doc_res):
    """
    get segement tag
    :param doc_res: 2-dim list
    :return: segement tag [0,0,0,1,1,2,2,2,2...]
    """
    seg_tag = []
    for index, paragraph in enumerate(doc_res):
        seg_tag += [index + 1] * len(paragraph)
    return seg_tag


def process_doc(doc_list, urlss):
    """
    split sentence and tokenize, get sentence seg tag
    :param doc_list: list of paragraphs
    :return: doc_sentence_list: tokenized
            seg_tag: seg tag of every sentence
    """
    doc_res = []  # 2-dim list
    urllist = []
    for i in range(len(doc_list)):
        paragraph = doc_list[i]
        _u = urlss[i]
        # split paragraph into sentences
        sentences = SentenceSplitter.split(paragraph)  # list of sentences
        # print(sentences)
        doc_res.append(sentences)
        urllist += [_u] * len(sentences)

    seg_tag = get_seg_tag(doc_res)
    doc_sentence_list = [" ".join(list(segmentor.segment(sent))) for p in doc_res for sent in
                         p]  # tokenized and 2-dim to 1-dim
    return doc_sentence_list, seg_tag, urllist

class Test():
    def __init__(self):
        if args.model == 'BERT_RNN':
            self.vocab = getattr(module, args.vocab)(args)
            net = getattr(models, args.model)(args)
        else:
            self.embed = torch.Tensor(np.load(args.embedding)['embedding'])
            # word2id = np.load(args.word2id)['word2id'].item()
            self.word2id = np.load(args.word2id, allow_pickle=True)['word2id'].item()
            self.vocab = getattr(module, args.vocab)(args, self.embed, self.word2id, use_gpu, args.use_elmo, args.use_seg,
                                                     args.use_mono)
            args.embed_num = self.embed.size(0)
            args.embed_dim = self.embed.size(1)

            # args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
            args.kernel_sizes = [3, 4, 5]
            self.net = getattr(models, args.model)(args)

        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

        if len(args.bert_gpu_ids) <= 1 and len(checkpoint['args'].bert_gpu_ids) > 1:
            new_od = OrderedDict()
            for k in checkpoint['model'].keys():
                new_od[re.sub('bert_model\.module\.', 'bert_model.', k)] = checkpoint['model'][k]
            checkpoint['model'] = new_od
        elif len(args.bert_gpu_ids) > 1 and len(checkpoint['args'].bert_gpu_ids) <= 1:
            new_od = OrderedDict()
            for k in checkpoint['model'].keys():
                new_od[re.sub('bert_model\.', 'bert_model.module.', k)] = checkpoint['model'][k]
            checkpoint['model'] = new_od

        self.net.load_state_dict(checkpoint['model'])
        # checkpoint['args']['device'] saves the device used as train time
        # if at test time, we are using a CPU, we must override device to None
        if not use_gpu:
            checkpoint['args'].device = None
    @torch.no_grad()
    def test(self, sentence_number, test_file):
        args.test_dir = test_file

        if args.debug:
            with codecs.open(args.test_dir, 'r', encoding='utf-8') as f:
                examples = [json.loads(line) for line in f][:2]
        else:
            with codecs.open(args.test_dir, 'r', encoding='utf-8') as f:
                # examples = [json.loads(line) for line in f]
                js = json.load(f)
                # print(js)
                examples = [js[key][0] for key in js]
                urls = [js[key][1] for key in js]
                # print(examples)
                # print(urls)
                doc_sentence_list, seg_tag, urlss = process_doc(examples, urls)
                seg_tag = [str(x) for x in seg_tag]
                # print(len(seg_tag))
                # print(len(doc_sentence_list))
                # print(len(urlss))

                examples = [{"doc": "\n".join(doc_sentence_list),
                            "segs": "\n".join(seg_tag),
                            "urls": "\n".join(urlss)}]
        # print(examples)
        test_dataset = module.Dataset(examples)

        test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        # net = getattr(models,checkpoint['args'].model)(checkpoint['args'])

        self.net.eval()
        doc_num = len(test_dataset)
        time_cost = 0
        file_id = 1
        result = []
        for batch in test_iter:
            # torch.cuda.empty_cache()
            # features,_,summaries,segs,mono,doc_lens,elmo = vocab.make_features(batch)
            features, segs, mono, doc_lens, elmo = self.vocab.make_test_features(batch)
            t1 = time()
            probs = self.net.test(Variable(features), Variable(segs), mono, doc_lens, elmo)

            para_probs = None
            if probs.dim() == 2:
                para_probs = probs[1]
                probs = probs[0]
            t2 = time()
            time_cost += t2 - t1
            start = 0
            for doc_id, doc_len in enumerate(doc_lens):
                # print(doc_len)
                stop = start + doc_len
                prob = probs[start:stop]
                if para_probs is not None:
                    para_prob = para_probs[start:stop]
                # topk = doc_len * args.topk
                topk = sentence_number  # 句子个数
                # ref = summaries[doc_id]
                # if args.bound > 0:
                #     bound_indices = [i for i, p in enumerate(prob) if p > args.bound]
                doc = batch['doc'][doc_id].split('\n')[:doc_len]
                # label = batch['labels'][doc_id].split('\n')[:doc_len]
                seg = batch['segs'][doc_id].split('\n')[:doc_len]
                url = batch['urls'][doc_id].split('\n')[:doc_len]
                assert len(seg) == len(prob)
                topk = int(min(topk, doc_len))
                # print(topk)
                topk_indices = prob.topk(topk)[1].cpu().data.numpy()
                topk_indices.sort()
                hyp = [doc[index] for index in topk_indices]  # 预测的topk个句子
                segg = [seg[index] for index in topk_indices]
                urll = [url[index] for index in topk_indices]
                tuples = [i for i in zip(segg, hyp, urll)]
                result.append(tuples)

                start = stop
                file_id = file_id + 1
                # print('Speed: %.2f docs / s' % (doc_num / time_cost))
        all_output = []
        i = 0
        # print(result)
        for one_doc in result:
            summary_results_example = one_doc
            postedit = PostEdit(summary_results_example, args.taskID)
            output = postedit.output
            # print(output)
            all_output.append(output)

            i += 1

        return all_output


if __name__ == '__main__':
    if args.test:
        tester = Test()
        tester.test()
