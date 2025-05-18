import pickle
from copy import deepcopy
import re
import itertools
from pathlib import Path
from typing import List, Tuple, Callable


import numpy as np
from Levenshtein import distance
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, LogitsProcessor
import datasets
from difflib import SequenceMatcher
from datasets import concatenate_datasets
from accelerate import PartialState
from accelerate.utils import gather_object


from utils.prepare_pdt import pdt_split_elements_to_tag, pdt_split_lemma_token, pdt_split_orth_token, pdt_tag_special_tokens


def get_tagset(tag_str):
    tagset = set()
    for tag in pdt_tag_special_tokens:
        if tag in tag_str:
            tagset.add(tag)
    return tagset


def split_element(element_string):
    element_dict = dict()
    temp = element_string.split(pdt_split_orth_token)
    assert len(temp) == 2
    element_dict['word'] = temp[0]
    temp = temp[1].split(pdt_split_lemma_token)
    assert len(temp) == 2
    element_dict['lemma'] = temp[0].lower()
    element_dict['tagset'] = get_tagset(temp[1])
    return element_dict


def reconstruct_word_list(split_elements):
    word_list = []
    for s_e in split_elements:
        word_list.append(s_e['word'])
    return word_list


def process_example(model_answer, answer, word_list, eos_token):
    if model_answer[-len(eos_token):] == eos_token:
        model_answer = model_answer[:-len(eos_token)]
        all_orths = True
    else:
        all_orths = False

    exact_example_match = model_answer == answer

    split_elements = answer.split(pdt_split_elements_to_tag)
    s_e_last = split_elements.pop()
    assert not s_e_last
    split_elements = [split_element(s_e) for s_e in split_elements]

    split_elements_model = model_answer.split(pdt_split_elements_to_tag)
    s_e_last = split_elements_model.pop()
    if all_orths:
        assert not s_e_last

    split_elements_model = [split_element(s_e) for s_e in split_elements_model]

    gold_word_list = reconstruct_word_list(split_elements)
    assert word_list == gold_word_list
    pred_word_list = reconstruct_word_list(split_elements_model)
    if all_orths:
        assert word_list == pred_word_list

    added_orths = 0
    if not all_orths:
        added_orths = (len(split_elements) - len(split_elements_model))
        split_elements_model = split_elements_model + ([None] * (len(split_elements) - len(split_elements_model)))

    assert len(split_elements) == len(split_elements_model)

    both_correct = {'tp': 0, 'fn': 0, 'fp': 0, 'correct': 0, 'wrong': 0}
    lemmas_correct = {'tp': 0, 'fn': 0, 'fp': 0, 'correct': 0, 'wrong': 0}
    tags_correct = {'tp': 0, 'fn': 0, 'fp': 0, 'correct': 0, 'wrong': 0}
    elements_count = len(split_elements)

    for s_e, s_e_m in zip(split_elements, split_elements_model):
        if s_e_m is None:
            lemmas_correct['fn'] += 1
            tags_correct['fn'] += 1
            both_correct['fn'] += 1
            lemmas_correct['wrong'] += 1
            tags_correct['wrong'] += 1
            both_correct['wrong'] += 1
        else:
            if s_e['lemma'] == s_e_m['lemma']:
                lemmas_correct['tp'] += 1
                lemmas_correct['correct'] += 1
                if s_e['tagset'] == s_e_m['tagset']:
                    tags_correct['tp'] += 1
                    both_correct['tp'] += 1
                    tags_correct['correct'] += 1
                    both_correct['correct'] += 1
                else:
                    tags_correct['fp'] += 1
                    tags_correct['fn'] += 1
                    both_correct['fp'] += 1
                    both_correct['fn'] += 1
                    tags_correct['wrong'] += 1
                    both_correct['wrong'] += 1
            elif s_e['tagset'] == s_e_m['tagset']:
                tags_correct['tp'] += 1
                lemmas_correct['fp'] += 1
                lemmas_correct['fn'] += 1
                both_correct['fp'] += 1
                both_correct['fn'] += 1
                tags_correct['correct'] += 1
                both_correct['wrong'] += 1
                lemmas_correct['wrong'] += 1
            else:
                lemmas_correct['fp'] += 1
                lemmas_correct['fn'] += 1
                tags_correct['fp'] += 1
                tags_correct['fn'] += 1
                both_correct['fp'] += 1
                both_correct['fn'] += 1
                tags_correct['wrong'] += 1
                both_correct['wrong'] += 1
                lemmas_correct['wrong'] += 1

    return int(exact_example_match), lemmas_correct, tags_correct, both_correct, elements_count, added_orths


class ForceAllLogitsProcessor(LogitsProcessor):
    def __init__(self, split_elements_to_tag_id, split_lemma_token_id, tag_token_ids, eos_token_id, split_orth_token_id, pre_answer_seq, tokenizer):
        self.pre_answer_seq = pre_answer_seq
        self.split_elements_to_tag_id = split_elements_to_tag_id
        self.split_lemma_token_id = split_lemma_token_id
        self.tag_token_ids = tag_token_ids
        self.eos_token_id = eos_token_id
        self.split_orth_token_id = split_orth_token_id
        self.after_tag_tokens_to_skip = None
        self.after_split_lemma_token_tokens_to_skip = None
        self.after_lemma_tokens_to_skip = None
        self.after_first_token_tokens_to_skip = None
        self.after_orth_split_token_tokens_to_skip = None
        self.after_orth_token_tokens_to_skip = None

        self.words_to_predict = None
        self.last_word_idx = None
        self.last_word_token_idx = None
        self.last_orth_token_idx = None
        self.leave_only_split_orth_token_id = None

        self.tokenizer = tokenizer
        self.leave_only_eos = None
        self.orth_ids = None

    def set_new_word_list_to_predict(self, words):
        self.words_to_predict = [[self.tokenizer(w, add_special_tokens=False)['input_ids'] for w in words_batch] for words_batch in words]
        self.curr_word_idx = [0] * len(words)
        self.curr_word_token_idx = [0] * len(words)
        return

    def remove_not_possible_orth_tokens_tensor(self, batch_scores, i):
        if self.curr_word_idx[i] >= len(self.words_to_predict[i]):
            if self.leave_only_eos is None:
                self.leave_only_eos = self.force_token(batch_scores, self.eos_token_id)
            batch_scores = self.leave_only_eos
        else:
            if self.curr_word_token_idx[i] >= len(self.words_to_predict[i][self.curr_word_idx[i]]):
                if self.leave_only_split_orth_token_id is None:
                    self.leave_only_split_orth_token_id = self.force_token(batch_scores, self.split_orth_token_id)
                batch_scores = self.leave_only_split_orth_token_id
                self.curr_word_token_idx[i] = 0
                self.curr_word_idx[i] += 1
            else:
                batch_scores = self.force_token(batch_scores, self.words_to_predict[i][self.curr_word_idx[i]][self.curr_word_token_idx[i]])
                self.curr_word_token_idx[i] += 1
        return batch_scores

    def process_removing_invalid_orth_tokens(self, batch_scores, i):
        batch_scores = self.remove_not_possible_orth_tokens_tensor(batch_scores, i)
        return batch_scores

    def force_token(self, batch_scores, token_id):
        leave_only_token = torch.full_like(batch_scores, -torch.inf)
        leave_only_token[token_id] = torch.inf
        return leave_only_token

    def __call__(self, input_ids, scores):
        new_scores = []
        for i, (batch_input_ids, batch_scores) in enumerate(zip(input_ids, scores)):
            last_token = batch_input_ids[-1].cpu().item()
            if last_token == self.split_lemma_token_id:
                if self.after_split_lemma_token_tokens_to_skip is None:
                    taggin_part_skip_tokens = list(range(scores.size()[1]))
                    for tag_token_id in self.tag_token_ids:
                        taggin_part_skip_tokens.remove(tag_token_id)
                    self.after_split_lemma_token_tokens_to_skip = torch.Tensor(taggin_part_skip_tokens).int().to(scores.device)
                batch_scores[self.after_split_lemma_token_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            elif last_token == self.split_orth_token_id:
                if self.after_orth_split_token_tokens_to_skip is None:
                    skip_tokens = deepcopy(self.tag_token_ids)
                    skip_tokens.append(self.split_elements_to_tag_id)
                    skip_tokens.append(self.eos_token_id)
                    skip_tokens.append(self.split_lemma_token_id)
                    skip_tokens.append(self.split_orth_token_id)
                    self.after_orth_split_token_tokens_to_skip = torch.Tensor(skip_tokens).int().to(scores.device)
                batch_scores[self.after_orth_split_token_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            elif last_token == self.split_elements_to_tag_id:
                batch_scores = self.process_removing_invalid_orth_tokens(batch_scores, i)
                new_scores.append(batch_scores)
            elif last_token in self.tag_token_ids:
                if self.after_tag_tokens_to_skip is None:
                    taggin_part_skip_tokens = list(range(scores.size()[1]))
                    for tag_token_id in self.tag_token_ids:
                        taggin_part_skip_tokens.remove(tag_token_id)
                    taggin_part_skip_tokens.remove(self.split_elements_to_tag_id)
                    self.after_tag_tokens_to_skip = torch.Tensor(taggin_part_skip_tokens).int().to(scores.device)
                batch_scores[self.after_tag_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            elif self.pre_answer_seq[-1] == last_token and self.pre_answer_seq == batch_input_ids[-len(self.pre_answer_seq):].tolist():
                batch_scores = self.process_removing_invalid_orth_tokens(batch_scores, i)
                new_scores.append(batch_scores)
            elif torch.sum(batch_input_ids == self.split_orth_token_id) <= torch.sum(batch_input_ids == self.split_lemma_token_id):
                batch_scores = self.process_removing_invalid_orth_tokens(batch_scores, i)
                new_scores.append(batch_scores)
            else:
                if self.after_lemma_tokens_to_skip is None:
                    skip_tokens = deepcopy(self.tag_token_ids)
                    skip_tokens.append(self.split_elements_to_tag_id)
                    skip_tokens.append(self.eos_token_id)
                    skip_tokens.append(self.split_orth_token_id)
                    self.after_lemma_tokens_to_skip = torch.Tensor(skip_tokens).int().to(scores.device)
                batch_scores[self.after_lemma_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
        return torch.stack(new_scores).to(scores.device)


def add_all_keys(sum_d, d):
    for key in sum_d.keys():
        sum_d[key] += d[key]
    return sum_d


def pdt_evaluate(accelerator, model, tokenizer, context_length, dataset_name, ds_dict_keys, batch_size, output_path, pickle_prefix=''):
    test_dataset = datasets.load_dataset(dataset_name)[ds_dict_keys]
    eos_token = tokenizer.eos_token
    split_elements_to_tag_id = tokenizer.convert_tokens_to_ids(pdt_split_elements_to_tag)
    split_lemma_token_id = tokenizer.convert_tokens_to_ids(pdt_split_lemma_token)
    split_orth_token_id = tokenizer.convert_tokens_to_ids(pdt_split_orth_token)
    tag_token_ids = []
    for t_t in pdt_tag_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(t_t)
        if token_id is not None:
            tag_token_ids.append(token_id)
    eos_token_id = tokenizer.eos_token_id
    for x in test_dataset:
        m = x['messages']
        m.pop()['content']
        prompt1 = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        prompt_tokenized1 = tokenizer(prompt1, add_special_tokens=False)
        prompt2 = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        prompt_tokenized2 = tokenizer(prompt2, add_special_tokens=False)
        break

    pre_answer_seq = prompt_tokenized1['input_ids'][len(prompt_tokenized2['input_ids']):]

    if not pre_answer_seq:
        pre_answer_seq = tokenizer.convert_tokens_to_ids('[/INST]')
        assert isinstance(pre_answer_seq, int)
        assert pre_answer_seq in tokenizer.all_special_ids
        pre_answer_seq = [pre_answer_seq]

    force_tokens_logits_processor = ForceAllLogitsProcessor(
        split_elements_to_tag_id, split_lemma_token_id, tag_token_ids, eos_token_id, split_orth_token_id, pre_answer_seq, tokenizer
    )

    logits_processor = LogitsProcessorList([force_tokens_logits_processor])

    pickle_filename = Path(output_path) / (pickle_prefix + '_results.pickle')

    if pickle_filename.exists():
        with open(pickle_filename, 'rb') as handle:
            result_list = pickle.load(handle)
    else:
        result_list = []

    batches = []
    j = 0
    batch = {'m': [], 'orginal_text': [], 'answer': [], 'words': []}
    for x in tqdm(test_dataset):
        j += 1
        if j <= len(result_list):
            continue
        m = x['messages']
        orginal_text = m[0]['content']
        orginal_text = re.sub('\s+', ' ', orginal_text)
        answer = m.pop()['content']
        words = x['word_list']
        batch['m'].append(m)
        batch['orginal_text'].append(orginal_text)
        batch['answer'].append(answer)
        batch['words'].append(words)

        if j < len(test_dataset) and len(batch['m']) < batch_size:
            continue

        prompt = tokenizer.apply_chat_template(batch['m'], tokenize=False, add_generation_prompt=True)
        prompt_tokenized = tokenizer(prompt, return_tensors='pt', add_special_tokens=False, padding='longest', padding_side='left')
        tokenized_prompt = prompt_tokenized['input_ids']
        max_new_tokens = context_length-tokenized_prompt.size()[1]
        input_lengths = tokenizer.batch_decode(tokenized_prompt)
        input_lengths = [i_l.replace(tokenizer.pad_token, '', -1) for i_l in input_lengths]
        prompt_lens = [len(i_l) for i_l in input_lengths]

        batch['max_new_tokens'] = max_new_tokens
        batch['tokenized_prompt'] = tokenized_prompt
        batch['prompt_lens'] = prompt_lens
        batch['prompt'] = prompt

        batches.append(batch)

        batch = {'m': [], 'orginal_text': [], 'answer': [], 'words': []}

    distributed_state = PartialState()
    model.to(distributed_state.device)
    with distributed_state.split_between_processes(batches) as sub_batches:
        for batch in tqdm(sub_batches):
            max_new_tokens = batch['max_new_tokens']
            tokenized_prompt = batch['tokenized_prompt']
            prompt_lens = batch['prompt_lens']
            prompt = batch['prompt']
            words = batch['words']
            tokenized_prompt = tokenized_prompt.to(distributed_state.device)
            logits_processor[0].set_new_word_list_to_predict(words)
            with torch.no_grad():
                output = model.generate(tokenized_prompt, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, logits_processor=logits_processor)
            model_answer = tokenizer.batch_decode(output)

            model_answer = [m_a.replace(tokenizer.pad_token, '', -1) for m_a in model_answer]
            model_answer = [m_a[p_l:] for m_a, p_l in zip(model_answer, prompt_lens)]

            for m_a, a, o_t, p, w in zip(model_answer, batch['answer'], batch['orginal_text'], prompt, words):
                exact_example_match, lemmas_correct, tags_correct, both_correct, elements_count, added_orths = process_example(m_a, a, word_list=w, eos_token=eos_token)
                result_list.append({
                    'answer': a,
                    'prompt': p,
                    'word_list': w,
                    'orginal_text': o_t,
                    'model_answer': m_a,
                    'exact_example_match': exact_example_match,
                    'lemmas_correct': lemmas_correct,
                    'tags_correct': tags_correct,
                    'both_correct': both_correct,
                    'elements_count': elements_count,
                    'added_orths': added_orths
                })

    result_list = gather_object(result_list)
    if result_list is not None and result_list:
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(result_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    exact_example_match = 0
    lemmas_correct = {'tp': 0, 'fn': 0, 'fp': 0, 'wrong': 0, 'correct': 0}
    tags_correct = {'tp': 0, 'fn': 0, 'fp': 0, 'wrong': 0, 'correct': 0}
    both_correct = {'tp': 0, 'fn': 0, 'fp': 0, 'wrong': 0, 'correct': 0}
    elements_count = 0
    added_orths = 0
    for x in result_list:
        exact_example_match += x['exact_example_match']
        lemmas_correct = add_all_keys(lemmas_correct, x['lemmas_correct'])
        tags_correct = add_all_keys(tags_correct, x['tags_correct'])
        both_correct = add_all_keys(both_correct, x['both_correct'])
        elements_count += x['elements_count']
        added_orths += x['added_orths']

    exact_example_match_score = exact_example_match * 100 / len(result_list)
    lemmas_correct_score = lemmas_correct['tp'] * 100 / (lemmas_correct['tp'] + 0.5 * (lemmas_correct['fp'] + lemmas_correct['fn']))
    tags_correct_score = tags_correct['tp'] * 100 / (tags_correct['tp'] + 0.5 * (tags_correct['fp'] + tags_correct['fn']))
    both_correct_score = both_correct['tp'] * 100 / (both_correct['tp'] + 0.5 * (both_correct['fp'] + both_correct['fn']))
    percentage_orths_skipped = added_orths * 100 / elements_count
    assert lemmas_correct['correct'] + lemmas_correct['wrong'] == elements_count
    lemmas_accuracy = lemmas_correct['correct'] * 100 / (lemmas_correct['correct'] + lemmas_correct['wrong'])
    assert tags_correct['correct'] + tags_correct['wrong'] == elements_count
    tags_accuracy = tags_correct['correct'] * 100 / (tags_correct['correct'] + tags_correct['wrong'])
    assert both_correct['correct'] + both_correct['wrong'] == elements_count
    both_accuracy = both_correct['correct'] * 100 / (both_correct['correct'] + both_correct['wrong'])

    print('exact_example_match', exact_example_match_score)
    print('lemmas f1', lemmas_correct_score)
    print('tags f1', tags_correct_score)
    print('both f1', both_correct_score)
    print('lemmas accuracy', lemmas_accuracy)
    print('tags accuracy', tags_accuracy)
    print('both accuracy', both_accuracy)
    print('elements count', elements_count)
    print('examples from datasaet count', len(result_list))
    print('skipped orths', percentage_orths_skipped, '%')

    score_dict = {
        'exact_example_match_score': exact_example_match_score,
        'lemmas_f1': lemmas_correct_score,
        'tags_f1': tags_correct_score,
        'both_f1': both_correct_score,
        'elements_count': elements_count,
        'examples_in_ds': len(result_list),
        'lemmas_acc': lemmas_accuracy,
        'tags_acc': tags_accuracy,
        'both_acc': both_accuracy,
        'percenrage_orth_skipped': percentage_orths_skipped
    }
    return score_dict
