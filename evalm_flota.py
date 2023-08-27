import pickle
import re

import torch
from transformers import AutoTokenizer


class FlotaTokenizer:
    def __init__(self, model, k, strict, mode):
        self.model = model
        self.k = k
        self.strict = strict
        self.mode = mode
        self.tok = AutoTokenizer.from_pretrained(model, model_max_length=128)#,local_files_only=True)
        self.vocab = self.tok.vocab.keys()
        assert len(self.vocab) == self.tok.vocab_size
        if self.model == 'google/muril-base-cased' or self.model == 'bert-base-multilingual-cased' or self.model == 'bert-base-multilingual-uncased':
            self.special = '##'
            self.max_len = 18
        elif self.model == 'gpt2':
            self.special = '\u0120'
            self.max_len = 19
        elif self.model == 'xlm-roberta-base':
            self.special = '▁'
            self.max_len = 18

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return len(self.vocab)
    def __call__(self, text):
        text = self.encode(text)
        batch_size = 1 #len(texts)
        max_len = 128#len(text)
        if self.model == 'google/muril-base-cased' or self.model == 'bert-base-multilingual-cased' or self.model == 'xlm-roberta-base':
            input_ids = torch.zeros(max_len).long()
            attention_mask = torch.zeros(max_len).long()
            token_type_id = torch.zeros(max_len).long()
            #for i, text in enumerate(texts):
            input_ids[ :len(text)] = torch.tensor(text)
            attention_mask[ :len(text)] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids':token_type_id}
            return tensors
        elif self.model == 'gpt2':
            input_ids = self.tok.eos_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
            return tensors
        elif self.model == 'xlnet-base-cased':
            input_ids = self.tok.pad_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
            tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
            return tensors

    def max_subword_split(self, w):
        for l in range(min(len(w), self.max_len), 0, -1):
            for i in range(0, len(w) - l + 1):
                if w[i] == '-':
                    continue
                subword = w[i:i + l]
                if self.model == 'google/muril-base-cased' or self.model == 'bert-base-multilingual-cased' or self.model == 'xlm-roberta-base':
                    if i == 0:
                        if subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                        elif not self.strict and self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                    else:
                        if self.special + subword in self. vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                        elif subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                elif self.model == 'gpt2' or self.model == 'xlnet-base-cased':
                    if i == 0:
                        if self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
                        elif subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                    else:
                        if subword in self.vocab:
                            return subword, w[:i] + l * '-' + w[i + l:], i
                        elif self.special + subword in self.vocab:
                            return self.special + subword, w[:i] + l * '-' + w[i + l:], i
        return None, None, None

    def get_flota_dict(self, w, k):
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return dict()
        if k == 1 or rest == len(rest) * '-':
            flota_dict = {i: max_subword}
            return flota_dict
        flota_dict = self.get_flota_dict(rest, k - 1)
        flota_dict[i] = max_subword
        return flota_dict

    def tokenize(self, w):
        if self.model == 'google/muril-base-cased' or self.model == 'bert-base-multilingual-cased' or self.model == 'xlm-roberta-base':
            if w in self.vocab:
                return [w]
            elif self.special + w in self.vocab:
                return [self.special + w]
        elif self.model == 'gpt2' or self.model == 'xlnet-base-cased':
            if self.special + w in self.vocab:
                return [self.special + w]
            elif w in self.vocab:
                return [w]
        if self.mode == 'flota':
            flota_dict = self.get_flota_dict(w, self.k)
            return [subword for i, subword in sorted(flota_dict.items())]
        elif self.mode == 'first':
            if self.model == 'gpt2':
                return self.tok.tokenize(' ' + w)[:self.k]
            return self.tok.tokenize(w)[:self.k]
        elif self.mode == 'longest':
            if self.model == 'gpt2':
                subwords = enumerate(self.tok.tokenize(' ' + w))
            else:
                subwords = enumerate(self.tok.tokenize(w))
            topk_subwords = sorted(subwords, key=lambda x: len(x[1].lstrip(self.special)), reverse=True)[:self.k]
            return [subword for i, subword in sorted(topk_subwords, key=lambda x: x[0])]

    def encode(self, text):
        text_split = text.split() #re.findall(r'[\w]+|[^\s\w]', text)
        tokens = list()
        for w in text_split:
            tokens.extend(self.tokenize(w))
        if self.model == 'google/muril-base-cased' or self.model == 'bert-base-multilingual-cased' or self.model == 'xlm-roberta-base':
            ids_flota = self.tok.convert_tokens_to_ids(tokens)[:self.tok.model_max_length - 2]
            return [self.tok.cls_token_id] + ids_flota + [self.tok.sep_token_id]
        elif self.model == 'gpt2':
            return self.tok.convert_tokens_to_ids(tokens)[:self.tok.model_max_length]
        elif self.model == 'xlnet-base-cased':
            ids_flota = self.tok.convert_tokens_to_ids(tokens)[:self.tok.model_max_length - 2]
            return ids_flota + [self.tok.sep_token_id, self.tok.cls_token_id]