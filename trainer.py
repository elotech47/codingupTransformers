import torch 
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path    
from dataset2 import TranslationDataset, causal_mask
from config import get_config, get_weights_path
from transformer import transformer_builder
import neptune
from tqdm import tqdm
import warnings
from dotenv import load_dotenv
import shutil
import torchmetrics
import argparse
import os

load_dotenv()

NEPTUNE_API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')

def initialize_logger(config):
    run = neptune.init_run(project=config['neptune_project'], api_token=NEPTUNE_API_TOKEN, name=config['experiment_name'], tags=config['tags'],
                           source_files=["*.py", "config.py", "dataset.py", "transformer.py", "transformer_trainer.py", "requirements.txt"])
    params = {k: v for k, v in config.items() if k not in ['neptune_project', 'experiment_name', 'tags']}
    run['parameters'] = params
    return run

def get_all_sentences(dataset, lang):
    for example in dataset:
        yield example['translation'][lang]

def create_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset = load_dataset('opus_books', f'{config["source_lang"]}-{config["target_lang"]}', split='train')
    
    src_tokenizer = create_tokenizer(config, dataset, config['source_lang'])
    tgt_tokenizer = create_tokenizer(config, dataset, config['target_lang'])
    
    # use 90% of the data for training and 10% for validation
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataset = TranslationDataset(train_dataset, src_tokenizer, tgt_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])
    val_dataset = TranslationDataset(val_dataset, src_tokenizer, tgt_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])
    
    max_len_src = max([len(src_tokenizer.encode(text).ids) for text in get_all_sentences(dataset, config['source_lang'])])
    max_len_tgt = max([len(tgt_tokenizer.encode(text).ids) for text in get_all_sentences(dataset, config['target_lang'])])
    
    print(f"Max source sequence length: {max_len_src}")
    print(f"Max target sequence length: {max_len_tgt}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, src_tokenizer, tgt_tokenizer

def create_model(config, src_vocab_size, tgt_vocab_size):
    return transformer_builder(
        src_vocab_size,
        tgt_vocab_size,
        d_model=config['d_model'],
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        num_layers=config['num_layers'],
        num_heads=config['nhead'],
        d_ff=config['dim_feedforward'],
        dropout=config['dropout']
    )

def greedy_decode(model, encoder_input, encoder_mask, tgt_tokenizer, max_len, device, print_msg):
    sos_idx = tgt_tokenizer.token_to_id("[SOS]")
    eos_idx = tgt_tokenizer.token_to_id("[EOS]")
    
    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.int64).to(device)
    
    for _ in range(max_len):
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        
        prob = model.generator(decoder_output[:, -1])
        _, next_token = torch.max(prob, dim=-1)
        next_token = next_token.item()
        
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], dtype=torch.int64).to(device)], dim=1)
        
        if next_token == eos_idx or decoder_input.size(1) == max_len:
            print_msg(f"Stopped decoding at step {decoder_input.size(1)}")
            break
        
    return decoder_input.squeeze(0)


def evaluate_model(model, validation_data, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_step=0, num_examples=2, logger=None):
    model.eval()
    count = 0
    
    source_texts = []
    target_texts = []
    predicted_texts = []
    
    columns = shutil.get_terminal_size().columns if shutil.get_terminal_size() else 80
    
    with torch.no_grad():
        for batch in validation_data:
            count += 1

            encoder_input = batch['encoder_input'].to(device) # [batch_size, seq_len]
            encoder_mask = batch['encoder_mask'].to(device) # [batch_size, seq_len]
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation"
            
            output = greedy_decode(model, encoder_input, encoder_mask, tgt_tokenizer, max_len, device, print_msg)
          
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            predicted_text = tgt_tokenizer.decode(output.cpu().numpy())
            
            source_texts.append(source_text)
            target_texts.append(target_text)
            predicted_texts.append(predicted_text)
            
            print_msg("-" * columns)
            print_msg(f"{f'Source: ':>12}{source_text}")
            print_msg(f"{f'Target: ':>12}{target_text}")
            print_msg(f"{f'Predicted: ':>12}{predicted_text}")
  
            if logger is not None:
                logger[f"Global step: {global_step}/Source text_{count}"] = source_text
                logger[f"Global step: {global_step}/Target text_{count}"] = target_text
                logger[f"Global step: {global_step}/Predicted text_{count}"] = predicted_text
                
            if count >= num_examples:
                print_msg("-" * columns)
                break
            
        # evaluate BLEU score and the character error rate
        metric = torchmetrics.BLEUScore()
        bleu_score = metric(predicted_texts, target_texts)
        if logger is not None:
            logger["metrics/bleu"].log(bleu_score)
        print_msg(f"BLEU score: {bleu_score}")  
        
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted_texts, target_texts)
        if logger is not None:
            logger["metrics/wer"].log(wer)
        print_msg(f"Word error rate: {wer}")
        
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted_texts, target_texts)
        if logger is not None:
            logger["metrics/cer"].log(cer)
        print_msg(f"Character error rate: {cer}")

def learning_rate_scheduler(optimizer, step, d_model, warmup_steps=4000):
    lr = (d_model ** -0.5) * min(step ** (-0.5), step * (warmup_steps ** -1.5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_model(config, logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    print("Using device:", device)
    if device == torch.device('cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == torch.device('mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, src_tokenizer, tgt_tokenizer = get_dataset(config)
    model = create_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size())
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config['preload_model'] is not None:
        model_filename = get_weights_path(config, config['preload_model'])
        print(f"Loading model from {model_filename}")
        checkpoint = torch.load(model_filename)
        initial_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['epochs']):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        batch_iterator = tqdm(train_loader, desc=f"Training epoch {epoch:02d}")
        for i, batch in enumerate(batch_iterator):
            optimizer.zero_grad()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            total_loss += loss.item()
            global_step += 1

            if i % 200 == 0:
                evaluate_model(model, val_loader, src_tokenizer, tgt_tokenizer, config['seq_len'], device, lambda x: batch_iterator.write(x), global_step=global_step, logger=logger)
            
            if logger is not None:
                logger["performance/train_loss"].log(loss.item())
                
            lr = learning_rate_scheduler(optimizer, global_step, config['d_model'])
            if logger is not None:
                logger["performance/lr"].log(lr)

        avg_loss = total_loss / len(train_loader)
        if logger is not None:
            logger["performance/avg_train_loss"].log(avg_loss)
        
        print(f"Epoch {epoch:02d} - Average loss: {avg_loss:6.3f}")
        
        evaluate_model(model, val_loader, src_tokenizer, tgt_tokenizer, config['seq_len'], device, lambda x: batch_iterator.write(x), global_step=global_step, logger=logger)
        
        model_filename = get_weights_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    
    if logger is not None:
        logger.stop()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a transformer model for translation")
    argparser.add_argument("--neptune_logger", action="store_true", help="Use neptune logger")
    
    args = argparser.parse_args()
    
    warnings.filterwarnings("ignore")
    config = get_config()
    logger = initialize_logger(config) if args.neptune_logger else None
    train_model(config, logger)
