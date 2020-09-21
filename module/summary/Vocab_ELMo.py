import torch
# from ELMoForManyLangs.elmoformanylangs import Embedder


class Vocab_ELMo():
    def __init__(self,args,embed,word2id,use_cuda=False,use_elmo=False,use_seg=False,use_mono=False):
        self.args = args
        self.sent_trunc = args.seq_trunc
        self.doc_trunc = args.pos_num
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v:k for k,v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1 
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.use_cuda = use_cuda
        self.use_elmo = use_elmo
        self.use_seg = use_seg
        self.use_mono = use_mono
        if use_elmo:
            self.elmo_path = args.elmo_path
            self.elmo_embedder = Embedder(args.elmo_path)
            self.ELMO_OOV_TOKEN = '<oov>'
            self.ELMO_PAD_TOKEN = '<pad>'

    def __len__(self):
        return len(word2id)

    def i2w(self,idx):
        return self.id2word[idx]
    def w2i(self,w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX
    
    def make_features(self,batch,sent_trunc=-1,doc_trunc=-1,split_token='\n'):
        if sent_trunc < 0:
            sent_trunc = self.sent_trunc
        if doc_trunc < 0:
            doc_trunc = self.doc_trunc
        sents_list,targets,doc_lens = [],[],[]
        segs = None
        monos = None
        if self.use_seg:
            segs = []
        if self.use_mono:
            monos = []
            # print(batch['role'])
        # trunc document 
        dtiter = zip(batch['doc'],batch['labels'])  
        for dt in dtiter:
            if self.use_seg and self.use_mono:
                doc,label,seg,mono = dt
            elif self.use_seg:
                doc,label,seg = dt
            elif self.use_mono:
                doc,label,mono = dt
            else:
                doc,label = dt
            sents = doc.split(split_token) # 因为有的\n\n之间有空格 
            labels = label.split(split_token)
            
            # if self.args.debug: 
            #     print(sents[:20]) 
            #     sents = [s.split() for s in sents]  
            #     print(sents[:20]) 
            #     sents = [' '.join(s) for s in sents if len(s)!= 0]  
            #     print(sents[:20]) 
                # input()
            # else:
            sents = [s.split() for s in sents]  
            sents = [' '.join(s) for s in sents if len(s)!= 0]  
            labels = [int(l) for l in labels] # eg:[0, 0,1,1,1 ,...] 
            # "segs": "1\n2\n3\n3\n3\n3\n4\n5\n6\n7\n7\n7\n7\n7\n7\n7\n7\n8\n8\n8\n8\n8\n8\n8\n8\n9\n10\n10\n11\n12\n13\n14"
            # "role": "0 2 3 4 5 7 25 28\n1 6\n8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 29 31\n30"
            if self.use_seg:
                pos = [int(p) for p in seg.split(split_token)]
            if self.use_mono: 
                # print(mono.type())
                # mono = [ids.split() for ids in mono.split('\n')]
                # mono = map(eval, mono)
                mono = eval(mono)
                # print(mono.type())
                # input()
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            if self.use_seg:
                pos = pos[:max_sent_num]
            sents_list += sents # sents [句子,juzi ]
            targets += labels # += 即 extend() '''不带[]'''
            if self.use_seg:
                segs += pos
            if self.use_mono:
                monos.append(mono)
                # print(segs)
                # print(monos)
                # input()

            doc_lens.append(len(sents)) # len(sents)=每一句的词个数 len(doc_lens)=句子个数
        # trunc or pad sent
        max_sent_len = 9
        batch_sents = []
        for sent in sents_list:     # sents_list=[sent, sent,...] 一个batch内的全部句子
            words = sent.split() 
            # if len(words) == 0:
            #     print('###############################')
            #     input()
            #     continue
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words) # batch_sents=[[w,w,w]..[w,w,w]] 一个batch内的全部句子
        
        features = []
        for sent in batch_sents: 
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)   # features=[[id,id,]..[id]] 一个batch内的全部句子
        
        elmo = None
        if self.use_elmo:
            with torch.no_grad():     # xy sz
                elmo = []
                for sent in batch_sents:
                    elmo.append(sent + [self.ELMO_PAD_TOKEN] * (max_sent_len-len(sent)))
                elmo = torch.FloatTensor(self.elmo_embedder.sents2elmo(elmo))

        features = torch.LongTensor(features)   # features[[id, id..], [id, id]] len(fea)=sent_num len(fea[0])=max_word_len
        targets = torch.LongTensor(targets)     # targets[1,0,0..1,0,0...]   
        summaries = batch['summaries']          # summaries # doc_lens[word_num, word_num]
        if self.use_seg:
            segs = torch.LongTensor(segs)
        if self.use_mono: 
            assert len(monos) == len(doc_lens)
            # print(monos)
            # print(segs)
            # monos = torch.LongTensor(monos)
            # input()

        if self.args.model == 'CNN_RNN_Multitask':
            para_targets = []
            sent_cnt = 0
            for doc_len in doc_lens:
                if doc_len > 0:
                    para_targets.append(1)
                for i in range(sent_cnt + 1, sent_cnt + doc_len):
                    para_targets.append(1 if segs[i] != segs[i - 1] else 0)
                sent_cnt += doc_len
            para_targets = torch.LongTensor(para_targets)
            targets = torch.cat([targets.unsqueeze(0), para_targets.unsqueeze(0)])

        return features,targets,summaries,segs,monos, doc_lens,elmo 
    
    def make_test_features(self,batch,sent_trunc=-1,doc_trunc=-1,split_token='\n'):
        if sent_trunc < 0:
            sent_trunc = self.sent_trunc
        if doc_trunc < 0:
            doc_trunc = self.doc_trunc
        sents_list, doc_lens = [], [] 
        segs = None
        monos = None
        if self.use_seg:
            segs = []
        if self.use_mono:
            monos = []
            # print(batch['role'])
        # trunc document
        # print('============')
        # print(batch)
        dtiter = zip(batch['doc'], batch['segs'])
        # print(dtiter)
        for dt in dtiter:
            # print(dt)
            doc, seg = dt
            sents = doc.split(split_token) # 因为有的\n\n之间有空格 
            # labels = label.split(split_token)
            # print(sents)
            sents = [s.split() for s in sents]  
            sents = [' '.join(s) for s in sents if len(s)!= 0]  
            # labels = [int(l) for l in labels] # eg:[0, 0,1,1,1 ,...] 
            # "segs": "1\n2\n3\n3\n3\n3\n4\n5\n6\n7\n7\n7\n7\n7\n7\n7\n7\n8\n8\n8\n8\n8\n8\n8\n8\n9\n10\n10\n11\n12\n13\n14"
            # "role": "0 2 3 4 5 7 25 28\n1 6\n8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 29 31\n30"
            if self.use_seg:
                pos = [int(p) for p in seg.split(split_token)] 
                # input()
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            # labels = labels[:max_sent_num]
            if self.use_seg:
                pos = pos[:max_sent_num]
            sents_list += sents # sents [句子,juzi ]
            # targets += labels # += 即 extend() '''不带[]'''
            if self.use_seg:
                segs += pos 

            doc_lens.append(len(sents)) # len(sents)=每一句的词个数 len(doc_lens)=句子个数
        # trunc or pad sent
        max_sent_len = 9
        batch_sents = []
        for sent in sents_list:     # sents_list=[sent, sent,...] 一个batch内的全部句子
            words = sent.split()  
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words) # batch_sents=[[w,w,w]..[w,w,w]] 一个batch内的全部句子
        
        features = []
        for sent in batch_sents: 
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)   # features=[[id,id,]..[id]] 一个batch内的全部句子
        
        elmo = None
        if self.use_elmo:
            with torch.no_grad():     # xy sz
                elmo = []
                for sent in batch_sents:
                    elmo.append(sent + [self.ELMO_PAD_TOKEN] * (max_sent_len-len(sent)))
                elmo = torch.FloatTensor(self.elmo_embedder.sents2elmo(elmo))

        features = torch.LongTensor(features)   # features[[id, id..], [id, id]] len(fea)=sent_num len(fea[0])=max_word_len
        # targets = torch.LongTensor(targets)     # targets[1,0,0..1,0,0...]   
        # summaries = batch['summaries']          # summaries # doc_lens[word_num, word_num]
        if self.use_seg:
            segs = torch.LongTensor(segs) 

        # if self.args.model == 'CNN_RNN_Multitask':
        #     para_targets = []
        #     sent_cnt = 0
        #     for doc_len in doc_lens:
        #         if doc_len > 0:
        #             para_targets.append(1)
        #         for i in range(sent_cnt + 1, sent_cnt + doc_len):
        #             para_targets.append(1 if segs[i] != segs[i - 1] else 0)
        #         sent_cnt += doc_len
        #     para_targets = torch.LongTensor(para_targets)
        #     targets = torch.cat([targets.unsqueeze(0), para_targets.unsqueeze(0)])

        return features, segs,monos, doc_lens,elmo
