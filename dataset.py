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
        self.sos_token_id = self.tgt_tokenizer.token_to_id("[CLS]")
        self.eos_token_id = self.tgt_tokenizer.token_to_id("[SEP]")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_target_pair = self.data[idx]
        src_text = src_target_pair['translation'][self.src_lang] if 'translation' in src_target_pair else src_target_pair[self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang] if 'translation' in src_target_pair else src_target_pair[self.tgt_lang]
        
        src_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids 
        
        # Ensure sequence length is large enough for tokens and special tokens
        if len(src_tokens) + 2 > self.seq_len or len(tgt_tokens) + 1 > self.seq_len:
            raise ValueError("Sequence length too small for the given input texts.")

        # Truncate tokens if necessary
        src_tokens = src_tokens[:self.seq_len - 2]
        tgt_tokens = tgt_tokens[:self.seq_len - 1]
        
        src_padding = [self.pad_token_id] * (self.seq_len - len(src_tokens) - 2)
        tgt_padding = [self.pad_token_id] * (self.seq_len - len(tgt_tokens) - 1)
        
        # Add SOS and EOS tokens to the source text
        encoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.int64),  # SOS token
            torch.tensor(src_tokens, dtype=torch.int64),  # Source tokens
            torch.tensor([self.eos_token_id], dtype=torch.int64),  # EOS token
            torch.tensor(src_padding, dtype=torch.int64)  # Padding
        ])
        
        decoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.int64),  # SOS token
            torch.tensor(tgt_tokens, dtype=torch.int64),  # Target tokens
            torch.tensor(tgt_padding, dtype=torch.int64)  # Padding
        ])
        
        label = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),  # Target tokens
            torch.tensor([self.eos_token_id], dtype=torch.int64),  # EOS token
            torch.tensor(tgt_padding, dtype=torch.int64)  # Padding
        ])
        
        assert len(encoder_input) == self.seq_len
        assert len(decoder_input) == self.seq_len
        assert len(label) == self.seq_len
        
        return {
            'encoder_input': encoder_input,
            'encoder_mask' : (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            'decoder_input': decoder_input,
            'decoder_mask': (decoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len, seq_len)
            'labels': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1).type(torch.int)  # Lower triangular matrix
    return mask  # (seq_len, seq_len)
