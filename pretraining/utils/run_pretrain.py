import logging
import math
import os
import sys
import subprocess
from typing import Optional
from functools import reduce

import datasets
import evaluate
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    set_verbosity_info,
    set_verbosity,
    enable_default_handler,
    enable_explicit_format
)


from utils.train_uilts import freeze_old_model_layers, add_new_layers
from utils.custom_tokenizer import load_tokenizer


logger = logging.getLogger(__name__)


def main(accelerator=None, preprocessed_datasets=None, cfg_primitive=None):
    training_args = cfg_primitive['training']
    custom_args = cfg_primitive['custom']

    training_args = TrainingArguments(**training_args)

    model_args = cfg_primitive['model']
    if 'trust_remote_code' not in model_args:
        model_args['trust_remote_code'] = False
    if 'model_revision' not in model_args:
        model_args['model_revision'] = 'main'
    if 'use_fast' not in model_args:
        model_args['use_fast'] = True
    if 'cache_dir' not in model_args:
        model_args['cache_dir'] = None
    if 'low_cpu_mem_usage' not in model_args:
        model_args = False

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.warning(
        f'Process rank: {training_args.local_rank}, '
        + f'device: {training_args.device}, n_gpu: {training_args.n_gpu}, '
        + 'distributed training: '
        + f'{training_args.parallel_mode.value == "distributed"},'
        + f' 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    last_checkpoint = None
    if (os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (last_checkpoint is None
                and len(os.listdir(training_args.output_dir)) > 0):
            raise ValueError(
                f'Output directory ({training_args.output_dir}) '
                + 'already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif (last_checkpoint is not None
              and training_args.resume_from_checkpoint is None):
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}.'
                + ' To avoid this behavior, change the `--output_dir` '
                + 'or add `--overwrite_output_dir` to train from scratch.'
            )

    set_seed(training_args.seed)

    tokenizer_kwargs = {
        'cache_dir': model_args['cache_dir'],
        'use_fast': model_args['use_fast_tokenizer'],
        'revision': model_args['model_revision'],
        'trust_remote_code': model_args['trust_remote_code'],
    }

    if custom_args['custom_tokenizer']:
        tokenizer = load_tokenizer(
            file=model_args['tokenizer_name'],
            fast=True,
            lang=custom_args['lang'],
            dictionary_path=custom_args['dictionary_path']
        )
    elif 'tokenizer_name' in model_args and model_args['tokenizer_name']:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args['tokenizer_name'],
            **tokenizer_kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args['model_name_or_path'],
            **tokenizer_kwargs
        )

    config_kwargs = {
        'cache_dir': model_args['cache_dir'],
        'revision': model_args['model_revision'],
        'trust_remote_code': model_args['trust_remote_code'],
    }

    config = AutoConfig.from_pretrained(
        model_args['model_name_or_path'],
        **config_kwargs
    )

    if custom_args['model_from_config']:
        config.max_position_embeddings = \
            cfg_primitive['data']['context_length']
        config.rope_scaling = None
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.vocab_size = len(tokenizer)
        config.is_decoder = True

    if custom_args['attention_mask_monkey_patch']:
        print('Document cross-attention masking')
        assert (model_args['attn_implementation'] == 'flash_attention_2')
        from utils.monkey_patch_packing import monkey_patch_packing_for_model
        monkey_patch_packing_for_model(model_args['model_name_or_path'])

    torch_dtype = (
        model_args['torch_dtype']
        if model_args['torch_dtype'] in ['auto', None]
        else getattr(torch, model_args['torch_dtype'])
    )

    if custom_args['model_from_config']:
        model = AutoModelForCausalLM.from_config(
            config=config,
            attn_implementation=model_args['attn_implementation'],
            torch_dtype=torch_dtype,
        )
        n_params = sum(
            {p.data_ptr(): p.numel() for p in model.parameters()}.values()
        )
        logger.info(
            'Training new model from scratch - '
            + f'Total size={n_params/2**20:.2f}M params'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args['model_name_or_path'],
            attn_implementation=model_args['attn_implementation'],
            from_tf=bool('.ckpt' in model_args['model_name_or_path']),
            config=config,
            cache_dir=model_args['cache_dir'],
            revision=model_args['model_revision'],
            trust_remote_code=model_args['trust_remote_code'],
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args['low_cpu_mem_usage'],
        )

    if custom_args['freeze_old_layers'] and not custom_args['add_new_layers']:
        model = freeze_old_model_layers(model, None)
    if custom_args['add_new_layers']:
        model = add_new_layers(
            model=model,
            state_dict_path=custom_args['new_state_dict_path'],
            freeze_old_layers=custom_args['freeze_old_layers']
        )
    if custom_args['add_padding_token']:
        tokenizer.pad_token = custom_args['new_pad_token']
        model.config.pad_token_id = tokenizer.pad_token_id

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    if custom_args['enable_input_require_grads']:
        model.enable_input_require_grads()

    if training_args.do_train:
        if 'train' not in preprocessed_datasets:
            raise ValueError('--do_train requires a train dataset')
        train_dataset = preprocessed_datasets['train']
    if training_args.do_eval:
        if 'validation' not in preprocessed_datasets:
            raise ValueError('--do_eval requires a validation dataset')
        eval_dataset = preprocessed_datasets['validation']

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load('accuracy', cache_dir=model_args['cache_dir'])

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    def save_to_s3():
        if custom_args['save_to_s3']:
            subprocess.run(
                [
                    'rclone',
                    'copy',
                    training_args.output_dir,
                    custom_args.s3_rclone_output_dir,
                    '--transfers=128',
                    '--s3-disable-checksum',
                    '--s3-upload-concurrency=32',
                    '--s3-chunk-size=128M'
                ]
            )
        return

    class OnSaveCallback(TrainerCallback):
        @accelerator.on_local_main_process
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
        ):
            save_to_s3()

    if custom_args['save_to_s3']:
        callbacks = [OnSaveCallback]
    else:
        callbacks = None

    if training_args.do_eval and not is_torch_xla_available():
        compute_metrics = compute_metrics
    else:
        compute_metrics = None

    if custom_args['custom_tokenizer']:
        class CustomTrainer(Trainer):
            def _save(self, output_dir: Optional[str] = None, state_dict=None):
                processing_class = self.processing_class
                self.processing_class = None
                super()._save(output_dir, state_dict)
                self.processing_class = processing_class
                return

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if (training_args.do_eval
                and not is_torch_xla_available()) else None,
            callbacks=callbacks
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if (training_args.do_eval
                and not is_torch_xla_available()) else None,
            callbacks=callbacks
        )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()

        metrics = train_result.metrics

        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

        save_to_s3()

    if training_args.do_eval:
        logger.info('*** Evaluate ***')

        metrics = trainer.evaluate()

        metrics['eval_samples'] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')
        metrics['perplexity'] = perplexity

        print('Perplexity', perplexity)

        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
