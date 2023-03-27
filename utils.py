import torch
import random
import numpy as np
import data
import youtokentome as yttm
import os
import importlib
from data.transforms import TextPreprocess


def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def remove_from_dict(the_dict, keys):
    copy_dict = dict(the_dict)
    for key in keys:
        copy_dict.pop(key, None)
    return copy_dict


def prepare_bpe(config):
    dataset_module = importlib.import_module(
        f'.{config.dataset.name}', data.__name__)
    # train BPE
    if config.bpe.get('train', False):
        dataset, ids = dataset_module.get_dataset(
            config, part='bpe', transforms=TextPreprocess())
        train_data_path = 'bpe_texts.txt'
        with open(train_data_path, "w", encoding='utf-8') as f:
            # run ovefr only train part
            for i in ids:
                text = dataset.get_text(i)
                f.write(f"{text}\n")
        yttm.BPE.train(data=train_data_path,
                       vocab_size=config.model.vocab_size, model=config.bpe.model_path)
        # os.system(f'rm {train_data_path}') #! TIN change

    bpe = yttm.BPE(model=config.bpe.model_path)
    return bpe
