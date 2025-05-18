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
    Trainer,
    AutoConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
import evaluate
import numpy as np
import datasets
from datasets import load_from_disk
from transformers.utils.logging import (
    set_verbosity_info,
    set_verbosity,
    enable_default_handler,
    enable_explicit_format
)
import torch
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live


from utils.custom_tokenizer import save_tokenize_fast
from utils.prepare_nkjp import prepare_tokenizer, create_compute_metrics_nkjp, prepare_penalizing_nkjp_tags
from utils.prepare_pdt import pdt_prepare_tokenizer, create_compute_metrics_pdt, prepare_penalizing_pdt_tags
from utils.prepare_tokenizer import get_tokenizer
from utils.prepare_dataset import create_sft_prepared_ds, get_ds_paths, load_datasplits

logger = logging.getLogger(__name__)


def main(
    accelerator=None,
    cfg_primitive=None,
    deepspeed_config=None,
):
    conf_training_args = cfg_primitive['training']
    custom_args = cfg_primitive['custom']
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
    if 'low_cpu_mem_usage' not in model_args:
        model_args['low_cpu_mem_usage'] = False

    lora_config = None
    if model_args['use_lora']:
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=model_args['lora_r'],
            lora_alpha=model_args['lora_alpha'],
            lora_dropout=model_args['lora_dropout'],
            bias=model_args['lora_bias'],
            task_type=model_args['lora_task_type'],
            )
        print(f'Use Lora training: {lora_config}')

    if custom_args['model_on_tmp_dir']:
        print(f'Add tmp dir prefix {os.environ["TMPDIR"]}')
        model_args['pretrained_model_name_or_path'] = os.environ['TMPDIR'] +\
            model_args['pretrained_model_name_or_path']

    set_seed(conf_training_args['seed'])

    torch_dtype = (
        model_args['torch_dtype']
        if model_args['torch_dtype'] in ['auto', None]
        else getattr(torch, model_args['torch_dtype'])
    )

    training_args = TrainingArguments(**conf_training_args, deepspeed=deepspeed_config)

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

    config_kwargs = {
        'cache_dir': model_args['cache_dir'],
        'revision': model_args['model_revision'],
        'trust_remote_code': model_args['trust_remote_code'],
    }

    config = AutoConfig.from_pretrained(
        model_args['pretrained_model_name_or_path'],
        **config_kwargs
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args['pretrained_model_name_or_path'],
        attn_implementation=model_args['attn_implementation'],
        from_tf=bool('.ckpt' in model_args['pretrained_model_name_or_path']),
        config=config,
        cache_dir=model_args['cache_dir'],
        revision=model_args['model_revision'],
        trust_remote_code=model_args['trust_remote_code'],
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args['low_cpu_mem_usage'],
    )

    if ('gradient_checkpointing' in conf_training_args
            and conf_training_args['gradient_checkpointing']):
        model.config.use_cache = False

    tokenizer, instruction_template, response_template = get_tokenizer(
        model_args=model_args,
        custom_args=custom_args,
    )

    if data_args['hf_dataset'] and 'nkjp' in data_args['dataset_special_type']:
        tokenizer = prepare_tokenizer(tokenizer)
    elif data_args['hf_dataset'] and data_args['dataset_special_type'] == 'pdt':
        tokenizer = pdt_prepare_tokenizer(tokenizer)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = eos_token_id

    new_train_path, new_eval_path, new_test_path = get_ds_paths(custom_args)

    @accelerator.on_local_main_process
    def process_datasets_local_main(
        tokenizer,
        custom_args,
        model_args,
        data_args,
        conf_training_args
    ):
        train_dataset, eval_dataset, test_dataset = load_datasplits(data_args)
        create_sft_prepared_ds(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            custom_args=custom_args,
            model_args=model_args,
            data_args=data_args,
            training_args=conf_training_args,
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
        )

    process_datasets_local_main(
        tokenizer,
        custom_args,
        model_args,
        data_args,
        conf_training_args
    )
    accelerator.wait_for_everyone()

    train_dataset = load_from_disk(new_train_path)
    if new_eval_path.exists():
        eval_dataset = load_from_disk(str(new_eval_path))
    else:
        eval_dataset = None
    if new_test_path.exists():
        test_dataset = load_from_disk(str(new_test_path))
    else:
        test_dataset = None

    if eval_dataset is None:
        training_args.eval_strategy = None
        training_args.do_eval = False

    def save_to_s3():
        if custom_args['save_to_s3']:
            subprocess.run(
                [
                    'rclone',
                    'copy',
                    training_args.output_dir,
                    custom_args['s3_rclone_output_dir'],
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

    def preprocess_logits_for_metrics(logits, labels):
        logits = logits[..., :-1, :].contiguous()
        logits = logits.argmax(dim=-1)
        return logits

    if data_args['dataset_special_type'] and 'nkjp' in data_args['dataset_special_type']:
        compute_metrics = create_compute_metrics_nkjp(tokenizer)
    elif data_args['dataset_special_type'] == 'pdt':
        compute_metrics = create_compute_metrics_pdt(tokenizer)
    else:
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            preds = torch.Tensor(preds).int()
            labels = torch.Tensor(labels[..., 1:]).contiguous().int()
            mask = labels != -100
            correct_predictions = (preds == labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            no_ignore_preds = preds.numpy()
            no_ignore_preds = np.where(
                no_ignore_preds != -100,
                no_ignore_preds,
                tokenizer.pad_token_id
            )
            decoded_preds = tokenizer.batch_decode(
                no_ignore_preds,
                skip_special_tokens=True
            )
            no_ignore_labels = labels.numpy()
            no_ignore_labels = np.where(
                no_ignore_labels != -100,
                no_ignore_labels,
                tokenizer.pad_token_id
            )
            decoded_labels = tokenizer.batch_decode(
                no_ignore_labels,
                skip_special_tokens=True,
            )
            decoded_preds = [
                ' '.join(pred.strip().split()) for pred in decoded_preds
            ]
            decoded_labels = [
                ' '.join(label.strip().split()) for label in decoded_labels
            ]
            rouge_metric_scorer = evaluate.load(
                'rouge',
                cache_dir=model_args['cache_dir'],
            )
            result = rouge_metric_scorer.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                rouge_types=['rouge1', 'rouge2']
            )
            accuracy = (correct_tokens.sum() / total_tokens.sum()).item() \
                if total_tokens.sum() > 0 else 0.0
            rows_are_equal = torch.eq(preds, labels).all(dim=1)
            exact_accuracy = rows_are_equal.float().mean()
            output = {k: round(v, 4) for k, v in result.items()}
            output['mean_token_accuracy'] = accuracy
            output['exact_accuracy'] = exact_accuracy
            return output

    if training_args.do_eval and not is_torch_xla_available():
        compute_metrics = compute_metrics
    else:
        compute_metrics = None

    if custom_args['enable_input_require_grads']:
        model.enable_input_require_grads()

    if 'eos_loss_penatly' in custom_args and custom_args['eos_loss_penatly'] or data_args['dataset_special_type'] == 'pdt':
        eos_loss_penatly = custom_args['eos_loss_penatly'] if 'eos_loss_penatly' in custom_args else 0
        assert isinstance(eos_loss_penatly, int) or isinstance(eos_loss_penatly, float)
        if eos_loss_penatly == 0 and not ('nkjp' in data_args['dataset_special_type'] and 'pos_tags_penatly' in custom_args and custom_args['pos_tags_penatly']):
            compute_loss_func = None
        else:
            def penalize_eos(loss, target):
                eos_mask = (target == eos_token_id)
                loss += eos_loss_penatly * (eos_mask.float() * loss)
                return loss

            penalize_functions = []

            if eos_loss_penatly != 0 and data_args['dataset_special_type'] != 'pdt':
                penalize_functions.append(penalize_eos)

            if ('nkjp' in data_args['dataset_special_type'] and 'pos_tags_penatly' in custom_args and custom_args['pos_tags_penatly']):
                penalize_nkjp_tags = prepare_penalizing_nkjp_tags(tokenizer)
                penalize_functions.append(penalize_nkjp_tags)
            elif data_args['dataset_special_type'] == 'pdt':
                if 'pos_tags_penatly' in custom_args and custom_args['pos_tags_penatly']:
                    penalize_pdt_tags = prepare_penalizing_pdt_tags(tokenizer)
                    penalize_functions.append(penalize_pdt_tags)

            def fixed_cross_entropy_penalties(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
                loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction='none')
                for p_f in penalize_functions:
                    loss = p_f(loss, target)
                if num_items_in_batch is not None:
                    loss = torch.sum(loss) / num_items_in_batch
                else:
                    loss = torch.mean(loss)
                return loss

            def ForCausalLMLoss(
                logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
            ):
                logits = logits.float()
                labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                shift_labels = labels[..., 1:].contiguous()
                logits = logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                loss = fixed_cross_entropy_penalties(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
                return loss

            def custom_loss(preditions, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False):
                loss = ForCausalLMLoss(
                    logits=preditions['logits'], labels=labels, vocab_size=vocab_size, num_items_in_batch=num_items_in_batch, disable_num_items_in_batch=disable_num_items_in_batch)
                return loss
            compute_loss_func = partial(custom_loss, vocab_size=model.config.vocab_size, disable_num_items_in_batch=False)
    else:
        compute_loss_func = None

    if not training_args.do_eval:
        preprocess_logits_for_metrics = None
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
            compute_loss_func=compute_loss_func,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if (training_args.do_eval
                and not is_torch_xla_available()) else None,
            compute_metrics=compute_metrics,
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
            compute_loss_func=compute_loss_func,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if (training_args.do_eval
                and not is_torch_xla_available()) else None,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )

    if training_args.do_train:
        trainer.train()
        trainer.save_model(training_args.output_dir)
        if custom_args['custom_tokenizer']:
            save_tokenize_fast(
                tokenizer,
                training_args.output_dir,
                lang=custom_args['lang']
            )
        else:
            tokenizer.save_pretrained(training_args.output_dir)
    if training_args.do_eval and test_dataset is not None:
        test_metrics = trainer.evaluate(test_dataset)
        print(test_metrics)


if __name__ == '__main__':
    main()
