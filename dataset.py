import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.data = dataset
        self.pad_token_id = self.src_tokenizer.token_to_id("[PAD]")
        self.sos_token_id = self.tgt_tokenizer.token_to_id("[SOS]")
        self.eos_token_id = self.tgt_tokenizer.token_to_id("[EOS]")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_target_pair = self.data[idx]
        src_text = src_target_pair['translation'][self.src_lang] if 'translation' in src_target_pair else src_target_pair[self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang] if 'translation' in src_target_pair else src_target_pair[self.tgt_lang]
        
        src_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids 
        
        # Truncate tokens if necessary
        src_tokens = src_tokens[:self.seq_len - 2]
        tgt_tokens = tgt_tokens[:self.seq_len - 1]
        
        src_padding = [self.pad_token_id] * (self.seq_len - len(src_tokens) - 2)
        tgt_padding = [self.pad_token_id] * (self.seq_len - len(tgt_tokens) - 1)
        
        # Add SOS and EOS tokens to the source text
        encoder_input = torch.tensor(
            [self.sos_token_id] + src_tokens + [self.eos_token_id] + src_padding,
            dtype=torch.int64
        )
        decoder_input = torch.tensor(
            [self.sos_token_id] + tgt_tokens + tgt_padding,
            dtype=torch.int64
        )
        label = torch.tensor(
            tgt_tokens + [self.eos_token_id] + tgt_padding,
            dtype=torch.int64
        )
        
        assert len(encoder_input) == self.seq_len
        assert len(decoder_input) == self.seq_len
        assert len(label) == self.seq_len
        
        encoder_mask = (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len).unsqueeze(0)
        
        return {
            'encoder_input': encoder_input,
            'encoder_mask': encoder_mask,
            'decoder_input': decoder_input,
            'decoder_mask': decoder_mask,
            'labels': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def causal_mask(seq_len):
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.int))  # Lower triangular matrix
    return mask  # (seq_len, seq_len)
