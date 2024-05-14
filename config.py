from pathlib import Path

def get_config():
    config = {
        'source_lang': 'en',
        'target_lang': 'fr',
        'batch_size': 4,
        'seq_len': 350,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'lr': 1e-4,
        'epochs': 10,
        'model_folder': 'models_weights',
        'model_fileame': 'transformermodel',
        "preload_model": None,
        "tokenizer_file": "tokenizers/tokenizers_{0}.json",
        "neptune_project": "DMO-LAB/llmfromscratch",
        "experiment_name": "simple_transformer",
        "tags": "transformer,translation,books,opus, attention",
        
    }
    
    return config

def get_weights_path(config, epoch):
    return Path(config['model_folder']) / f"{config['model_filename']}_epoch_{epoch}.pt"

def get_latest_weights(config):
    weights = list(Path(config['model_folder']).glob(f"{config['model_filename']}_epoch_*.pt"))
    if len(weights) == 0:
        return None
    return max(weights, key=lambda x: x.stat().st_ctime)