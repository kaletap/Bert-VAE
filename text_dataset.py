import random
from typing import List

import nltk
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_sequence_length: int, sent_tokenize: bool = True, min_sent_length: int = 4):

        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.sent_tokenize = sent_tokenize
        self.min_sent_length = min_sent_length
        self.w2i = tokenizer.get_vocab()
        self.i2w = {idx: word for word, idx in self.w2i.items()}
        self.cls_token = tokenizer.cls_token
        self.eos_token = tokenizer.eos_token or tokenizer.sep_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text_all = self.dataset[idx]['text']
        sentences = [sent for sent in nltk.tokenize.sent_tokenize(text_all) if len(sent) > self.min_sent_length]
        text = random.choice(sentences) if sentences else text_all
        tokens = self.tokenizer.tokenize(text)[:self.max_sequence_length - 1]

        input_tokens = [self.cls_token] + tokens
        target_tokens = tokens + [self.eos_token]

        input = self.tokenizer.convert_tokens_to_ids(input_tokens)
        target = self.tokenizer.convert_tokens_to_ids(target_tokens)

        assert len(input) == len(target), "%i, %i" % (len(input), len(target))
        length = len(input)
        return {
            'input': input,
            'target': target,
            'length': length
        }

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def pad_idx(self):
        return self.tokenizer.pad_token_id

    @property
    def sos_idx(self):
        return self.tokenizer.cls_token_id

    @property
    def eos_idx(self):
        return self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: List[dict]):
        length = [example['length'] for example in examples]
        biggest_length = max(length)
        # we assume that Dataset takes care of truncation
        input_ids = [example['input'] + [self.pad_token_id]*(biggest_length - len(example['input'])) for example in examples]
        target_ids = [example['target'] + [self.pad_token_id]*(biggest_length - len(example['input'])) for example in examples]

        input_ids = torch.tensor(input_ids)
        target_ids = torch.tensor(target_ids)
        length = torch.tensor(length)

        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            'input': input_ids,
            'target': target_ids,
            'attention_mask': attention_mask,
            'length': length
        }

