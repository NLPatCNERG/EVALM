from collections import defaultdict
from transformers import AutoTokenizer,XLMRobertaTokenizer
import re,sys
import numpy as np
from scipy.stats import entropy
import math
import pandas as pd
model_name = sys.argv[1]
input_train_file = sys.argv[2]
w_c_threshold = int(sys.argv[3])
output_vocab_file = sys.argv[4]
desired_vocab_count = int(sys.argv[5])
trans_file_path = sys.argv[6]
ent_threshold = int(sys.argv[7])
start = 0x0900 #0x0C00(te) #0x0B80(ta) #0x0D00(ml) # 0x0A80(gu)#0x0980(bn) #0x0900(hi)   #https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html
end = 0x097F #0x0C7F(te) #0x0BFF(ta) #0x0DFF(ml) #0x0AFF(gu)#0x09ff(bn) #0x097f(hi)
unicode_re = re.compile(r'^(##)*[\U{0:08X}-\U{1:08X}\d]*'
                         r'[\U{0:08X}-\U{1:08X}]+'
                         r'[\U{0:08X}-\U{1:08}\d]*$'.format(start, end))
tokenizer = AutoTokenizer.from_pretrained(model_name)#, local_files_only=True)


token_count_dic = defaultdict(int)
token_word_dic = defaultdict(set)
word_count_dic = defaultdict(int)
word_count_rel_wise_dic = defaultdict(int)
token_count_rel_wise_dic = defaultdict(int)
word_entropy_dic = {}
token_entropy_dic = {}
trans_dic = {}
with open(trans_file_path,'r') as rd:
    while True:
        line = rd.readline()
        if line == '':
            break
        try:
            bn_word,en_word = line.strip().split('\t')
        except:
            continue
        trans_dic[bn_word] = en_word
#df_train = pd.read_csv(input_train_file,names=['Data','Label'],skiprows=1)
df_train = pd.read_csv(input_train_file,names=['Label','Data'],header=None) #need to be changed according to data format
#print(df_train.head())
label_list = list(df_train.Label.unique())
for idx,row in df_train.iterrows():
    sen = row['Data']  
    rel = row['Label']
    word_list = sen.strip().split()
    for w in word_list:
      if unicode_re.match(w):
        token_list = tokenizer.tokenize(w)
        word_count_dic[w] += 1
        word_count_rel_wise_dic[(w,rel)] += 1
        for t in token_list:
          token_count_dic[t] += 1
          token_count_rel_wise_dic[(t,rel)] += 1
          token_word_dic[t].add(w)  

for w in word_count_dic.keys():
  tmp_list = []
  final_list = []
  tmp_count = 0
  for rel in label_list:
    tmp_list.append(word_count_rel_wise_dic[(w,rel)])
    tmp_count += word_count_rel_wise_dic[(w,rel)]   
  final_list = [t/tmp_count for t in tmp_list]
  word_entropy_dic[w] = (round(entropy(final_list),3),tmp_list)  

for t in token_count_dic.keys():
  tmp_count = 0
  tmp_list = []
  final_list = []
  for rel in label_list:
    tmp_list.append(token_count_rel_wise_dic[(t,rel)])
    tmp_count += token_count_rel_wise_dic[(t,rel)]
  final_list = [t/tmp_count for t in tmp_list]    
  token_entropy_dic[t] = (round(entropy(final_list),3),tmp_list,tmp_count)


tmp_dic = {}
for w in word_count_dic.keys():
    token_list = tokenizer.tokenize(w)
    token_list_len = len(token_list)
    total_entropy = 0
    for t in token_list:
        total_entropy += token_entropy_dic[t][0]/token_list_len #*len(t.replace('##','')))/len(w) #token_list_len
    tmp_dic[w] = [word_count_dic[w],(token_list_len/len(w))*100,word_entropy_dic[w][0],round(total_entropy,3),round(((total_entropy+0.0001-word_entropy_dic[w][0])/(total_entropy+0.0001))*100,2)] 
    
with open(output_vocab_file,'w') as wr:
    count = 0
    for item in sorted(tmp_dic.items(), key=lambda k: (k[1][4]),reverse=True):
        if item[1][0] >= w_c_threshold and item[1][4] >= ent_threshold:
            if item[0] in trans_dic:
                wr.write(item[0]+'\t'+trans_dic[item[0]]+'\n')
            else:
                wr.write(item[0]+'\t'+item[0]+'\n')
            count += 1
            if count >= desired_vocab_count:
                break