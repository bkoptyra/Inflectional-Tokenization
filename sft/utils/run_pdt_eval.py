import json
import sys
import torch
from pathlib import Path
from typing import Optional
import os
import subprocess
import logging
from functools import partial
import gc


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

import torch


from utils.evaluation_pdt import pdt_evaluate
from utils.custom_tokenizer import get_pretokenizer


def main(
    accelerator=None,
    cfg_primitive=None,
):
    custom_args = cfg_primitive['custom']
    conf_training_args = cfg_primitive['training']
    data_args = cfg_primitive['data']
    model_args = cfg_primitive['model']
    if 'trust_remote_code' not in model_args:
        model_args['trust_remote_code'] = False
    if 'model_revision' not in model_args:
        model_args['model_revision'] = 'main'
    if 'use_fast' not in model_args:
        model_args['use_fast'] = True
    if 'cache_dir' not in model_args:
        model_args['cache_dir'] = None

    torch_dtype = (
        model_args['torch_dtype']
        if model_args['torch_dtype'] in ['auto', None]
        else getattr(torch, model_args['torch_dtype'])
    )

    config_kwargs = {
        'cache_dir': model_args['cache_dir'],
        'revision': model_args['model_revision'],
        'trust_remote_code': model_args['trust_remote_code'],
    }

    config = AutoConfig.from_pretrained(
        conf_training_args['output_dir'],
        **config_kwargs
    )

    model = AutoModelForCausalLM.from_pretrained(
        conf_training_args['output_dir'],
        attn_implementation=model_args['attn_implementation'],
        from_tf=bool('.ckpt' in conf_training_args['output_dir']),
        config=config,
        cache_dir=model_args['cache_dir'],
        revision=model_args['model_revision'],
        trust_remote_code=model_args['trust_remote_code'],
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(conf_training_args['output_dir'])
    if custom_args['custom_tokenizer'] and custom_args['lang'] in ['pl_lcs', 'pl_suffix', 'cz_lcs']:
        pretokenizer = get_pretokenizer(lang=custom_args['lang'], dictionary_path=custom_args['dictionary_path'])
        tokenizer._tokenizer.pre_tokenizer = pretokenizer
    model.eval()
    if 'multiple_seeds' in custom_args and custom_args['multiple_seeds']:
        prefix = custom_args['prefix']
    else:
        prefix = 'eval'
    scores = pdt_evaluate(accelerator, model, tokenizer, model_args['model_max_length'], data_args['train_data_path'], 'test', int(custom_args['inference_batch_size']), conf_training_args['output_dir'], pickle_prefix=prefix)
    print(scores)
    accelerator.log(scores)


if __name__ == '__main__':
    main()
