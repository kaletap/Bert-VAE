from typing import List

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_sequence_length):

        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.w2i = tokenizer.get_vocab()
        self.i2w = {idx: word for word, idx in self.w2i.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        tokens = self.tokenizer.tokenize(text)[:self.max_sequence_length - 1]

        input_tokens = [self.tokenizer.cls_token] + tokens
        target_tokens = tokens + ['<eos>']

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
        return self.tokenizer.convert_tokens_to_ids('<pad>')

    @property
    def sos_idx(self):
        return self.w2i['<cls>']

    @property
    def eos_idx(self):
        return self.w2i['<pad>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def __call__(self, examples: List[dict]):
        length = [example['length'] for example in examples]
        biggest_length = max(length)
        # we assume that Dataset takes care of truncation
        input_ids = [example['input'] + [self.pad_idx]*(biggest_length - len(example['input'])) for example in examples]
        target_ids = [example['target'] + [self.pad_idx]*(biggest_length - len(example['input'])) for example in examples]

        input_ids = torch.tensor(input_ids)
        target_ids = torch.tensor(target_ids)
        length = torch.tensor(length)

        attention_mask = (input_ids != self.pad_idx).long()

        return {
            'input': input_ids,
            'target': target_ids,
            'attention_mask': attention_mask,
            'length': length
        }

