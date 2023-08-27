import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AlbertForSequenceClassification, AutoTokenizer, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, BertModel
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import sys
import logging
from sklearn.metrics import f1_score 
from typing import List
import json
import os
from collections import defaultdict
import random
from evalm_flota import *
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

gpu = sys.argv[1]
train_file = sys.argv[2]
val_file = sys.argv[3]
test_file = sys.argv[4]
save_model_file = sys.argv[5]
save_pred_file = sys.argv[6]
SEED = int(sys.argv[7])
patience_threshold = int(sys.argv[9]) 
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
model_name = sys.argv[8]#'google/muril-base-cased' #'bert-base-multilingual-cased'
LR = 2e-5
max_epoch = 0
EPOCHS = 15
max_sen_len = 128
train_batch_size = 16
val_batch_size = 128
test_batch_size = 128

#tokenizer = AutoTokenizer.from_pretrained(model_name)#model_name,local_files_only=True)
tokenizer = FlotaTokenizer(model_name, 3, True, 'flota') #for flota baseline

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['Label']]
        self.texts_mod = [tokenizer(text) for text in df['Data']] #for flota baseline
        #self.texts_mod = [tokenizer(text, 
        #                       padding='max_length', max_length = max_sen_len, truncation=True,
        #                        return_tensors="pt",return_token_type_ids=True) for text in df['Data']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts_mod(self, idx):
        # Fetch a batch of inputs
        return self.texts_mod[idx]   

    def get_batch_sen_id(self, idx):
        # Fetch a batch of inputs
        return self.sen_id[idx]    

    def __getitem__(self, idx):

        batch_texts_mod = self.get_batch_texts_mod(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts_mod, batch_y

class BertClassifier(nn.Module):

    def __init__(self,class_number,model_name):

        super(BertClassifier, self).__init__()

        if 'indic' in model_name: 
            self.bert = AlbertForSequenceClassification.from_pretrained(model_name,num_labels = class_number,   
                output_attentions = False,
                output_hidden_states = True)
        elif 'roberta' in model_name:
            self.bert = XLMRobertaForSequenceClassification.from_pretrained(model_name,num_labels = class_number,   
                output_attentions = False,
                output_hidden_states = True)#,local_files_only=True)
        else:    
            self.bert = BertForSequenceClassification.from_pretrained(model_name,num_labels = class_number,   
                output_attentions = False,
                output_hidden_states = True)#,local_files_only=True)

    def forward(self, input_id, mask, token_type_id, labels):

        loss, logits, hidden_states = self.bert(input_ids= input_id, attention_mask=mask, token_type_ids=token_type_id,labels=labels,return_dict=False)

        return loss, logits,hidden_states

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=val_batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device(gpu if use_cuda else "cpu")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-08)

    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    if use_cuda:
            model = model.to(device)
            #criterion = criterion.to(device)
    model.zero_grad()        
    max_acc = 0
    patience_count = 0
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            train_pred_labels = None
            train_gold_labels = None

            train_ids = None
            train_logits = None
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].squeeze(1).to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                token_type_id = train_input['token_type_ids'].squeeze(1).to(device)

                model.train()
                loss,logits,hidden_states_up = model(input_id, mask,token_type_id,train_label)
                total_loss_train += loss.item()
                batch_loss = loss
                output = logits 
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                if train_pred_labels is None:
                    train_pred_labels = output.argmax(dim=1).detach().cpu().numpy()
                    train_gold_labels = train_label.detach().cpu().numpy()
                else:
                    train_pred_labels = np.append(train_pred_labels, output.argmax(dim=1).detach().cpu().numpy(), axis=0)
                    train_gold_labels = np.append(train_gold_labels, train_label.detach().cpu().numpy(), axis=0)

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            train_f1_score = f1_score(train_gold_labels, train_pred_labels, average='macro')
            total_acc_val = 0
            total_loss_val = 0
            valid_pred_labels = None
            valid_gold_labels = None

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].squeeze(1).to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    token_type_id = val_input['token_type_ids'].squeeze(1).to(device)

                    batch_loss,logits,hidden_states_up = model(input_id, mask, token_type_id, val_label)
                    total_loss_val += batch_loss.item()
                    output = logits
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    if valid_pred_labels is None:
                        valid_pred_labels = output.argmax(dim=1).detach().cpu().numpy()
                        valid_gold_labels = val_label.detach().cpu().numpy()
                    else:
                        valid_pred_labels = np.append(valid_pred_labels, output.argmax(dim=1).detach().cpu().numpy(), axis=0)
                        valid_gold_labels = np.append(valid_gold_labels, val_label.detach().cpu().numpy(), axis=0)
                
                valid_f1_score = f1_score(valid_gold_labels, valid_pred_labels, average='macro')
                
                if valid_f1_score > max_acc:
                    if valid_f1_score - max_acc > 0.001:
                        patience_count = 0
                    else:
                        patience_count += 1    
                    max_acc = valid_f1_score
                    torch.save(model.state_dict(), save_model_file)  
                    print('Model saved at Epoch:',epoch_num+1)
                else:
                    patience_count += 1    
                if patience_count == patience_threshold:
                    print("No improvment for last 5 epoch...breaking...")
                    break          
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f}| Train f1_score: {train_f1_score: .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}| Val f1_score: {valid_f1_score: .3f}')

def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=test_batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device(gpu if use_cuda else "cpu")
    model.load_state_dict(torch.load(save_model_file))
    model.eval()

    if use_cuda:

        model = model.to(device)

    total_acc_test = 0
    test_pred_labels = None
    test_gold_labels = None
    with torch.no_grad(),open(save_pred_file,'w') as wr:

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].squeeze(1).to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              token_type_id = test_input['token_type_ids'].squeeze(1).to(device)

              _,output_mod,_ = model(input_id, mask, token_type_id, test_label)
            
              output = output_mod 
              acc = (output.argmax(dim=1) == test_label).sum().item()
              acc_score = torch.max(output,dim=1)[0]
              for idx,item in enumerate(output.argmax(dim=1)):
                wr.write(str(item.item())+'\t'+str(test_label[idx].item())+'\t'+str(acc_score[idx].item())+'\t'+str(output[idx])+'\n')
              total_acc_test += acc
              if test_pred_labels is None:
                test_pred_labels = output.argmax(dim=1).detach().cpu().numpy()
                test_gold_labels = test_label.detach().cpu().numpy()
              else:
                test_pred_labels = np.append(test_pred_labels, output.argmax(dim=1).detach().cpu().numpy(), axis=0)
                test_gold_labels = np.append(test_gold_labels, test_label.detach().cpu().numpy(), axis=0)
                
        test_f1_score = f1_score(test_gold_labels, test_pred_labels, average='macro')
    
    print("Seed:: ",SEED)
    print(f'Test Accuracy:: {total_acc_test / len(test_data): .3f}')
    print(f'Test f1_score(weighted):: {test_f1_score: .3f}')  

if 'gu-train' in train_file or 'ml-train' in train_file or 'hi-train' in train_file:
    if 'addon' in train_file:
        df_train = pd.read_csv(train_file,names=['Label','Data'],header=None)
    else:
        df_train = pd.read_csv(train_file,names=['Label','Data'],header=None)
    df_val = pd.read_csv(val_file,names=['Label','Data'],header=None)
    df_test = pd.read_csv(test_file,names=['Label','Data'],header=None) 

elif 'gluecos_sentiment_en-hi_train' in train_file:
    if 'addon' in train_file:
        df_train = pd.read_csv(train_file,names=['Data','Label'],header=None)
    else:
        df_train = pd.read_csv(train_file,names=['Data','Label'],sep='\t',header=None)
    df_val = pd.read_csv(val_file,names=['Data','Label'],sep='\t',header=None)
    df_test = pd.read_csv(test_file,names=['Data','Label'],sep='\t',header=None)

elif 'sentnob_train' in train_file:
    if 'addon' in train_file:
        df_train = pd.read_csv(train_file,names=['Data','Label'],header=None)
    else:    
        df_train = pd.read_csv(train_file,names=['Data','Label'],skiprows=1)
    df_val = pd.read_csv(val_file,names=['Data','Label'],skiprows=1)
    df_test = pd.read_csv(test_file,names=['Data','Label'],skiprows=1)

elif 'bn_hate_speech_train' in train_file:
    if 'addon' in train_file:
        df_train = pd.read_csv(train_file,names=['Index','Label','Data'],header=None)
    else:    
        df_train = pd.read_csv(train_file,names=['Index','Label','Data'],skiprows=1,sep='\t')
    df_val = pd.read_csv(val_file,names=['Index','Label','Data'],skiprows=1,sep='\t')
    df_test = pd.read_csv(test_file,names=['Index','Label','Data'],skiprows=1,sep='\t')

epoch_wise_sen_attention = defaultdict(lambda: defaultdict(list))
labels = {}
rev_labels = {}
for idx,item in enumerate(list(df_train.Label.unique())):
  labels[item] = idx
  rev_labels[idx] = item
model = BertClassifier(len(labels),model_name)
train(model, df_train, df_val, LR, EPOCHS)
evaluate(model, df_test)
