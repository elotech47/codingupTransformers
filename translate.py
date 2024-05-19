import torch 
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformer import transformer_builder
from pathlib import Path    
from dataset import TranslationDataset, causal_mask
from config import get_config, get_weights_path, get_latest_weights
from transformer import transformer_builder
import warnings
import torchmetrics
import os
from transformer_trainer import create_model, create_tokenizer, greedy_decode, get_dataset

def translate(sentence: str, config):
    # Load the latest model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)
    print(f"Using device: {device}")
    
    config = get_config()
    dataset = load_dataset(f"{config['datasource']}", f"{config['source_lang']}-{config['target_lang']}", split='all')
    src_tokenizer = create_tokenizer(config, dataset, config['source_lang'])
    tgt_tokenizer = create_tokenizer(config, dataset, config['target_lang'])
    
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        dataset = TranslationDataset(dataset, src_tokenizer, tgt_tokenizer, src_lang=config['source_lang'], tgt_lang=config['target_lang'], seq_len=config['seq_len'])
        sentence = dataset[id]['src_text']
        label = dataset[id]["tgt_text"]
    seq_len = config['seq_len']
    
    model_name = get_latest_weights(config)
    print(f"Loading model from {model_name}")
    model_state = torch.load(model_name, map_location=device)
    
    model = transformer_builder(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), d_model=config['d_model'], num_heads=config['nhead'], num_layers=config['num_layers'], d_ff=config['dim_feedforward'], dropout=config['dropout'], src_seq_len=seq_len, tgt_seq_len=seq_len)
    model.load_state_dict(model_state['model_state_dict'])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        pad_token_id = src_tokenizer.token_to_id("[PAD]")
        sos_token_id = tgt_tokenizer.token_to_id("[SOS]")
        eos_token_id = tgt_tokenizer.token_to_id("[EOS]")
        print(pad_token_id, sos_token_id, eos_token_id)
        source = src_tokenizer.encode(sentence)
        source = torch.tensor([sos_token_id] + source.ids + [eos_token_id] + [pad_token_id] * (seq_len - len(source.ids) - 2)).unsqueeze(0).to(device)
        source_mask = (source != src_tokenizer.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tgt_tokenizer.token_to_id('[SOS]')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')
        
         # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # # print the translated word
            print(f"{tgt_tokenizer.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tgt_tokenizer.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tgt_tokenizer.decode(decoder_input[0].tolist())
    
    

if __name__ == "__main__":
    config = get_config()
    sentence = "How are you?."
    translate(sentence, config)