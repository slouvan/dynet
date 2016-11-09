
# coding: utf-8

# In[1]:

class TrainingDataReader:
    def __init__(self):
        self.paragraph = []

    def read_paragraph(self, file_name):
        with open(file_name,"r") as fp:
            for line in fp:
                processed_line = line.replace("\\newline"," ")
                para = ParagraphData()
                data = processed_line.split("\t")
                story = data[2]
                para.story = story
                for i in range(3,len(data),5):
                    question = QuestionData()
                    question_text = ""
                    if data[i].startswith("multiple: ") :
                        question_text = data[i].replace("multiple: ","")
                    else:
                        question_text = data[i].replace("one: ","")
                    
                    question.question_text = question_text
                    for choice_idx in range(i+1, i+5):
                        question.choices.append(data[choice_idx])
                    para.questions.append(question)
                self.paragraph.append(para)
    
    def print_training_data(self):
        print len(self.paragraph)
        for i in range(0, len(self.paragraph)):
            print str(self.paragraph[i])
    
    def read_answer(self, file_name):
        with open(file_name, "r") as fp:
            p_idx = 0
            for line in fp:
                correct_answers = line.split("\t")
                for q_idx in range(0, len(correct_answers)):
                    #self.paragraph[p_idx].questions[q_idx].correct_answer = self.paragraph[p_idx].questions[q_idx].choices[correct_answers-'A']
                    self.paragraph[p_idx].questions[q_idx].correct_answer = self.paragraph[p_idx].questions[q_idx].choices[ord(list(correct_answers[q_idx])[0])-ord('A')]

                    #print "CORRECT: ",self.paragraph[p_idx].questions[q_idx].correct_answer
                p_idx += 1
    
    def construct_training_instances(self):
        # paragraph, question, answer, label
        training_instances = []
        for i in range(0, len(self.paragraph)):
            para_obj = self.paragraph[i]
            story_txt = para_obj.story
            for j in range(0, len(para_obj.questions)):
                question_obj = para_obj.questions[j]
                question_txt = question_obj.question_text
                for k in range(0, len(question_obj.choices)):
                    question_choice_txt = question_obj.choices[k]
                    if question_choice_txt == question_obj.correct_answer:
                        training_instances.append((story_txt,question_txt, question_choice_txt, 1))
                    else:
                        training_instances.append((story_txt,question_txt, question_choice_txt, 0))
        return training_instances

class ParagraphData:
    def __init__(self):
        self.story = ""
        self.questions = []
    def __str__( self ):
        str_rep = ""
        str_rep += self.story+"\n"
        for i in range(0, len(self.questions)):
            str_rep += "Question "+str(i)+" "+str(self.questions[i])
        return str_rep

class QuestionData:
    def __init__(self):
        self.question_text = ""
        self.choices = []
        self.correct_answer = ""
        
    def __str__( self ):
        str_rep = ""
        str_rep += self.question_text+"\n"
        for i in range(0, len(self.choices)):
            if self.choices[i] == self.correct_answer:
                str_rep += self.choices[i]+"(CORRECT ANSWER)\n"
            else:
                str_rep += self.choices[i]+"\n"
        return str_rep

def main():
    reader = TrainingDataReader()
    reader.read_paragraph("/home/slouvan/dynet/data/MCTest/mc160.train.tsv")
    reader.read_answer("/home/slouvan/dynet/data/MCTest/mc160.train.ans")
    #reader.print_training_data()
    t_instances = reader.construct_training_instances()
    print len(t_instances)
    


# In[ ]:

# IMPORT
from collections import Counter, defaultdict
from itertools import count
import random
from nltk.tokenize import word_tokenize as w_tokenizer
from nltk.tokenize import sent_tokenize as s_tokenizer

import dynet as dy
import numpy as np

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def collect_sentences(t_instances):
    sents = {}
    for i in range(len(t_instances)):
        story_txt, question_txt, answer_choice, label = t_instances[i]
        if story_txt not in sents:
            sents[story_txt] = 0
        if question_txt not in sents:
            sents[question_txt] = 0
        if answer_choice not in sents:
            sents[question_txt] = 0
    
    return list(sents.keys())

data_reader = TrainingDataReader()
data_reader.read_paragraph("/home/slouvan/dynet/data/MCTest/mc160.train.tsv")
data_reader.read_answer("/home/slouvan/dynet/data/MCTest/mc160.train.ans")
t_instances = data_reader.construct_training_instances()

train_sentences = collect_sentences(t_instances) # might contain paragraph, so we need to break the sentence
words = []
wc    = Counter()

for data in train_sentences:
    sents = s_tokenizer(data)
    for sent in sents:
        tokens = w_tokenizer(sent)
        for token in tokens:
            words.append(token)
            wc[token]+=1

words.append("__UNK__")
wc["__UNK__"]+=1
vw = Vocab.from_corpus([words])
nwords = vw.size()


# DYNET STUFF
# DyNet Starts
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Lookup parameters for word embeddings
WORDS_LOOKUP = model.add_lookup_parameters((nwords, 64))


RNN   = dy.LSTMBuilder(1, 64, 64, model)
RNN_2 = dy.LSTMBuilder(1, 64, 64, model)



pW = model.add_parameters((32, 64)) # atau 128?
pV = model.add_parameters((32))
pB = model.add_parameters(32)

def token_lookup(w):
    if w in vw.w2i :
        return WORDS_LOOKUP[vw.w2i[w]]
    else:
        return WORDS_LOOKUP[vw.w2i["__UNK__"]]

def sent_rep(sent):
    tokens = w_tokenizer(sent)
    tokens_rep_list = []
    for token in tokens:
        tokens_rep_list.append(token_lookup(token))
    
    return dy.average(tokens_rep_list)

def token_rep(sent):
    #print sent
    tokens = w_tokenizer(sent)
    tokens_rep_list = []
    for token in tokens:
        tokens_rep_list.append(token_lookup(token))
    
    return tokens_rep_list

def calc_loss(para, question_txt, answer_choice, correct_label):
    dy.renew_cg()
    f_init = RNN.initial_state()
    g_init = RNN.initial_state()
    
    # break a paragraph into sentences
    sents = s_tokenizer(para)
    s_exprs = []
    closs = 0
    sent_rep_list = []
    # construct sentences representation
    for sent in sents:
        sent_rep_list.append(sent_rep(sent))
    
    question_answer = question_txt+" "+answer_choice # IS it possible to have > 1 sent in a question, need to fix this
    #question_answer_rep = sent_rep(question_answer)
    
    # Construct the dummy "tree", for paragraph
    
    s = f_init
    stack_paragraph = []
    counter = 0
    stack_paragraph.append(s)
    
    
    while len(stack_paragraph) > 0 and counter < len(sent_rep_list):
        if counter == 0:
            composed_vec = dy.average([sent_rep_list[counter], sent_rep_list[counter + 1]])
            s = stack_paragraph.pop()
            s = s.add_input(composed_vec)
            stack_paragraph.append(s.output())
            counter += 2
            #print "Masuk"
            #exit()
        else:
            prev_hidden = stack_paragraph.pop()
            s = s.add_input(dy.average([prev_hidden, sent_rep_list[counter] ]))
            counter += 1
            stack_paragraph.append(s.output())
    
    para_exp = stack_paragraph.pop()
    
    q = g_init
    stack_question_answer = []
    stack_question_answer.append(q)
    token_rep_list = token_rep(question_answer)
    counter = 0
    while len(stack_question_answer) > 0 and counter < len(token_rep_list):
        if counter == 0:
            composed_vec = dy.average([token_rep_list[counter], token_rep_list[counter + 1]])
            q = stack_question_answer.pop()
            q = q.add_input(composed_vec)
            stack_question_answer.append(q.output())
            counter += 2
        else:
            prev_hidden = stack_question_answer.pop()
            q = q.add_input(dy.average([prev_hidden, token_rep_list[counter] ]))
            counter += 1
            stack_question_answer.append(q.output())
    
    q_a_exp = stack_question_answer.pop()
    
    W = dy.parameter(pW)
    b = dy.parameter(pB)
    v = dy.parameter(pV)
    
    x = dy.average([para_exp,q_a_exp]) # Compose paragraph + question answer
    # predict
    yhat = dy.logistic(dy.dot_product(v,dy.tanh(W*x+b)))
    #loss
    
    if correct_label == 0:
        loss = -dy.log(1 - yhat)
    elif correct_label == 1:
        loss = -dy.log(yhat) 
    
    closs += loss.scalar_value() #forward
    loss.backward()
    trainer.update()
    return closs

for epoch in range(0,100):
    for t_instance in t_instances:
        closs = 0
        closs += calc_loss(t_instance[0], t_instance[1], t_instance[2], t_instance[3])

    print closs





