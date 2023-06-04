# func.py

import re
from konlpy.tag import Mecab
from transformers import AutoTokenizer
import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
import qna
import model 

mecab = Mecab()
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
device = model.device

# 0 번호 1 장소 2 시간 3 공부
stopwords_tag = [
    'IC',
    'EP', 'EF', 'EC', 'ETN', 'ETM', #어미
    'MAG', 'MAJ',#일반부사
    'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ','JX', 'JC', #조사들
    'SF','SE','SSO','SSC','SC', 'SY',
    'SL','SH' ,'SN'#외국어, 한자 ,숫자
]

intent_dic = {
  0 : qna.df_num,
  1 : qna.df_where,
  2 : qna.df_when,
  3 : qna.df_study
}

def text_preprocess(text,mecab=mecab,stopwords_tag=stopwords_tag):
  pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ]+')
  text = re.sub(pattern, '', text)
  pp_text = ""

  for n,pos in mecab.pos(text):
    if all(pos in stopwords_tag for pos in pos.split('+')):
        continue
    if pos in stopwords_tag:
        continue
    pp_text += n
  
  return pp_text
    
def tokenizing(text,tokenizer=tokenizer):
  tk = tokenizer(
      text, 
      return_tensors='pt',
      truncation=True,
      max_length=256,
      padding = "max_length",
      add_special_tokens=True
  ).to(device)
  return tk

def intent(model,text):
  token = tokenizing(text)
  return torch.argmax(model(**token).logits).item()

def text_embedding(pp_text):
  tk = tokenizing(pp_text)
  return tk['input_ids'].cpu().numpy()[0]


def jaccard_sim(text,keywords,mecab=mecab,num_intent=False):
  k_list = keywords.split(',')
  if num_intent==True:
    count = 0
    for i in k_list:
      if i in text:
        count += 1
    return count/len(k_list)

  t_list = mecab.morphs(text)
  union = set(k_list).union(set(t_list))
  intersection = set(k_list).intersection(set(t_list))
  
  return len(intersection)/len(union)

def cos_sim(A, B):
  return dot(A, B.T)/(norm(A)*norm(B))

def find_answer(text, intent):
  df = intent_dic[intent]

  pp_text = text_preprocess(text)
  emb = text_embedding(pp_text)
  tf = intent==2 or intent==3

  jarc = []
  cos = []
  for idx,i in enumerate(df.values):
    comp_emb = list(map(int, (i[3])[1:-1].split()))
    jarc.append(jaccard_sim(text,i[0],mecab, tf))
    cos.append(cos_sim(np.array(emb),np.array(comp_emb)))

  if sum(jarc)==0:
    return "Unkown answer"

  res = np.array(jarc)*np.array(cos) 
  answer_idx = res.argmax()
  
  return df.iloc[answer_idx]['answer']