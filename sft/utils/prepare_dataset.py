from pathlib import Path
import os
import json


import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, concatenate_datasets, load_from_disk
from trl import DataCollatorForCompletionOnlyLM


from utils.prepare_tokenizer import get_tokenizer
from utils.prepare_pdt import mask_orth_and_eos_tokens


config_path = "conf_sft"
config_name = "base-config"


def _load_dialog_json(raw_data: str):
    data = json.loads(raw_data)['fillers']
    output = []
    for message in data:
        output.append({'role': message['role'], 'content': message['content']})
    del data
    return output


def _load_flat_json(raw_data: str):
    data = json.loads(raw_data)
    if data.get('final_answer') is None:
        output = []
        for message in data['fillers']:
            output.append(
                {'role': message['role'], 'content': message['content']}
            )
        return output
    if data.get('final_input') is None:
        data['final_input'] = data['filler_input']
    return [
        {'role': 'user', 'content': str(data['final_input'])},
        {'role': 'assistant', 'content': str(data['final_answer'])}
    ]


def enroll_conversations(example):
    enrolled_convs = []
    other_keys = [key for key in example.keys() if key != "messages"]
    other_keys_dict = {key: [] for key in other_keys}
    for conv in example["messages"]:
        start_id = 2
        # if conversation have 'system prompt' enroll will start from 3rd index
        if conv[0]["role"] == "system":
            start_id = 3
        for i in range(start_id, len(conv) + 2, 2):
            enrolled_convs.append(conv[:i])
            for key in other_keys:
                other_keys_dict[key].append(example[key])
    converted_data = {key: other_keys_dict[key] for key in other_keys}
    converted_data["messages"] = enrolled_convs
    return converted_data


def get_ds_paths(custom_args):
    new_train_path = 'train_ds'
    new_eval_path = Path('eval_ds')
    new_test_path = Path('test_ds')
    if custom_args['dataset_temp_on_tmp_dir']:
        new_train_path = str(Path(os.environ['TMPDIR']) / new_train_path)
        new_eval_path = Path(os.environ['TMPDIR']) / new_eval_path
        new_test_path = Path(os.environ['TMPDIR']) / new_test_path
    return new_train_path, new_eval_path, new_test_path


@hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name
)
def main(cfg: DictConfig):
    cfg_primitive = OmegaConf.to_container(cfg)
    training_args = cfg_primitive['training']
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
        model_args = False
    create_sft_prepared_ds(
        custom_args=custom_args,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    return


def get_train_data_path(data_args):
    if data_args['s3_data']:
        train_data_path = \
            Path(str(os.environ['TMPDIR'])) / data_args['train_data_path']
    else:
        train_data_path = Path(data_args['train_data_path'])
    print(f'Load arrow dataset from {train_data_path}')
    return train_data_path


def load_datasplits(data_args):
    eval_dataset = None
    test_dataset = None

    train_data_path = get_train_data_path(data_args)

    if data_args['hf_dataset']:
        ds_dict = load_dataset(str(train_data_path))
        if data_args['dataset_special_type'] == 'nkjp_cv':
            cv_test_split = data_args.get('cv_test_split', None)
            assert cv_test_split is not None
            train_splits = list(ds_dict.keys())
            train_splits.remove(cv_test_split)
            train_dataset = concatenate_datasets([ds_dict[k] for k in train_splits])
            test_dataset = ds_dict[cv_test_split]
        else:
            train_dataset = ds_dict['train']
            if 'val' in ds_dict:
                eval_dataset = ds_dict['val']
            if 'test' in ds_dict:
                test_dataset = ds_dict['test']
    else:
        datasets_to_use = dict()
        possible_subsets = [
            'curated_datasets',
            'original_datasets',
            'external_datasets',
            'synthetic_datasets',
        ]

        training_subfolders = data_args['training_subfolders']
        for possible_subset in possible_subsets:
            if possible_subset in data_args:
                datasets_to_use[possible_subset] = \
                    data_args[possible_subset]

        train_datasets = []

        for folder, subfolder in datasets_to_use.items():
            train_datasets.append(
                load_from_disk(
                    str(
                        train_data_path / folder /
                        subfolder / training_subfolders
                    )
                )
            )
        if 'rag_data' in data_args:
            train_datasets.append(
                load_dataset(
                    'json',
                    data_files={
                        'train': str(
                            train_data_path / 'rag_data' /
                            data_args['rag_data'] / 'train.jsonl'
                        )
                    }
                )
            )

        train_dataset = concatenate_datasets(
            [t['train'] for t in train_datasets]
        )
        eval_datasets = []
        test_datasets = []
        for ds in train_datasets:
            if 'val' in ds:
                eval_datasets.append(ds['val'])
            if 'test' in ds:
                test_datasets.append(ds['test'])
        if eval_datasets:
            eval_dataset = concatenate_datasets(
                eval_datasets
            )
        if test_datasets:
            test_dataset = concatenate_datasets(
                test_datasets
            )
    return train_dataset, eval_dataset, test_dataset


def create_sft_prepared_ds(
        train_dataset,
        eval_dataset,
        test_dataset,
        custom_args,
        model_args,
        data_args,
        training_args,
        tokenizer=None,
        instruction_template=None,
        response_template=None,
):
    train_dataset = train_dataset.shuffle(seed=training_args['data_seed'])
    if data_args['enroll_conversations']:
        print('Enrolling conversations!')
        train_dataset = train_dataset.select_columns(['messages'])
        train_dataset = train_dataset.map(
            enroll_conversations,
            batched=True,
            num_proc=custom_args['num_processes']
        )
    print(train_dataset)
    train_dataset = train_dataset.filter(
        lambda example: len(example['messages']) > 1
    )
    print(train_dataset)

    if tokenizer is None:
        tokenizer, instruction_template, response_template = get_tokenizer(
            model_args,
            custom_args
        )

    print(train_dataset)

    if custom_args['completion_only']:
        def filter_row(examples):
            texts = [
                tokenizer.apply_chat_template(
                    example,
                    tokenize=False,
                )
                for example in examples['messages']
            ]
            tokenized = tokenizer(
                texts,
                padding=False,
                truncation=False,
                max_length=model_args['model_max_length'],
                return_attention_mask=True,
            )
            return [len(t) <= int(model_args['model_max_length'])
                    for t in tokenized['input_ids']]

        train_dataset = train_dataset.filter(
            filter_row,
            batched=True
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(
                filter_row,
                batched=True
            )
        if test_dataset is not None:
            test_dataset = test_dataset.filter(
                filter_row,
                batched=True
            )

    print(train_dataset)

    if instruction_template is not None:
        instruction_template_ids = tokenizer.encode(
                instruction_template, add_special_tokens=False,
        )
    if response_template is not None:
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False,
        )

    if custom_args['completion_only']:
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template_ids,
            response_template=response_template_ids,
            tokenizer=tokenizer,
            mlm=False,
            return_tensors='pt'
        )

    def tokenize(examples):
        texts = [
            tokenizer.apply_chat_template(
                example,
                tokenize=False,
            )
            for example in examples['messages']
        ]
        if ('padding_type' in data_args['padding_type'] and
                data_args['padding_type'] is not None):
            padding = data_args['padding_type']
            if padding is not None:
                return_tensors = 'pt'
            else:
                return_tensors = None
        else:
            if custom_args['completion_only']:
                padding = 'max_length'
                return_tensors = 'pt'
            else:
                padding = False
                return_tensors = None
        tokenized = tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=model_args['model_max_length'],
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        if custom_args['completion_only']:
            data_for_collator = []
            for ids, a_m in zip(
                tokenized['input_ids'],
                tokenized['attention_mask']
            ):
                data_for_collator.append(
                    {
                        'input_ids': ids,
                        'attention_mask': a_m,
                        'labels': ids
                    }
                )
            tokenized = collator(data_for_collator)
        examples['input_ids'] = tokenized['input_ids']
        if 'labels' in tokenized:
            examples['labels'] = tokenized['labels']
        else:
            examples['labels'] = tokenized['input_ids']
            examples['labels'][examples['labels'] == tokenizer.pad_token_id] \
                = -100
        if data_args['dataset_special_type'] == 'pdt':
           examples['labels'] = [mask_orth_and_eos_tokens(label_row, tokenizer) for label_row in examples['labels']]
        examples['attention_mask'] = tokenized['attention_mask']
        return examples

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    if test_dataset is not None:
        test_dataset = test_dataset.map(
            tokenize,
            batched=True,
            remove_columns=test_dataset.column_names,
        )

    new_train_path, new_eval_path, new_test_path = get_ds_paths(
        custom_args=custom_args
    )

    train_dataset.save_to_disk(
        new_train_path,
        num_proc=custom_args['num_processes']
    )
    if eval_dataset is not None:
        eval_dataset.save_to_disk(
            str(new_eval_path),
            num_proc=custom_args['num_processes']
        )
    if test_dataset is not None:
        test_dataset.save_to_disk(
            str(new_test_path),
            num_proc=custom_args['num_processes']
        )


if __name__ == '__main__':
    main()
