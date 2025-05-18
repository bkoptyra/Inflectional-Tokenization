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


from utils.prepare_nkjp import split_elements_to_tag, split_lemma_token, nps_token, split_orth_token, tag_special_tokens


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def is_valid_utf8(byte_list):
    try:
        bytes(byte_list).decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def align_lists_distance(
    gold: List, pred: List, distance: Callable[[object, object], float]
) -> Tuple[List, List]:
    LARGE_NUM = 100_000_000 
    m, n = len(gold), len(pred)

    dp = np.full((m + 1, n + 1), float('inf'))
    dp[0][0] = 0

    for i in range(m + 1):
        for j in range(n + 1):
            if i > 0 and j > 0:
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + distance(gold[i - 1], pred[j - 1]))
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + LARGE_NUM)
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + LARGE_NUM)

    aligned_gold, aligned_pred = [], []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + distance(gold[i - 1], pred[j - 1]):
            aligned_gold.append(gold[i - 1])
            aligned_pred.append(pred[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + LARGE_NUM:
            aligned_gold.append(gold[i - 1])
            aligned_pred.append(None)
            i -= 1
        else:
            aligned_gold.append(None)
            aligned_pred.append(pred[j - 1])
            j -= 1

    return aligned_gold[::-1], aligned_pred[::-1]


def align_lists_with_nones(source, target):
    matcher = SequenceMatcher(None, source, target)
    aligned_source = []
    aligned_target = []

    src_index = 0
    tgt_index = 0

    for match in matcher.get_matching_blocks():

        if src_index < match.a or tgt_index < match.b:
            src_slice = source[src_index:match.a]
            tgt_slice = target[tgt_index:match.b]
            src_index += 1
            source_slice_aligned_list, target_slice_aligne_list = align_lists_distance(src_slice, tgt_slice, distance)
            for source_slice_aligned, target_slice_aligned in zip(source_slice_aligned_list, target_slice_aligne_list):
                aligned_source.append(source_slice_aligned)
                aligned_target.append(target_slice_aligned)
            src_index = match.a
            tgt_index = match.b

        for _ in range(match.size):
            aligned_source.append(source[src_index])
            aligned_target.append(target[tgt_index])
            src_index += 1
            tgt_index += 1

    return aligned_source, aligned_target


def get_tagset(tag_str):
    tagset = set()
    for tag in tag_special_tokens:
        if tag in tag_str:
            tagset.add(tag)
    return tagset


def add_all_keys(sum_d, d):
    for key in sum_d.keys():
        sum_d[key] += d[key]
    return sum_d


class ForceOrderLogitsProcessor(LogitsProcessor):
    def __init__(self, split_elements_to_tag_id, split_lemma_token_id, tag_token_ids, eos_token_id, split_orth_token_id, nps_token_id, pre_answer_seq):
        self.pre_answer_seq = pre_answer_seq
        self.split_elements_to_tag_id = split_elements_to_tag_id
        self.split_lemma_token_id = split_lemma_token_id
        self.tag_token_ids = tag_token_ids
        self.eos_token_id = eos_token_id
        self.nps_token_id = nps_token_id
        self.split_orth_token_id = split_orth_token_id
        self.after_tag_tokens_to_skip = None
        self.after_split_lemma_token_tokens_to_skip = None
        self.after_split_elements_to_tag_id_tokens_to_skip= None
        self.after_lemma_tokens_to_skip = None
        self.after_first_token_tokens_to_skip = None
        self.after_nps_token_tokens_to_skip = None
        self.after_orth_split_token_tokens_to_skip = None
        self.after_orth_token_tokens_to_skip = None

    def __call__(self, input_ids, scores):
        new_scores = []
        for batch_input_ids, batch_scores in zip(input_ids, scores):
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
            elif last_token == self.nps_token_id:
                if self.after_nps_token_tokens_to_skip is None:
                    skip_tokens = deepcopy(self.tag_token_ids)
                    skip_tokens.append(self.split_elements_to_tag_id)
                    skip_tokens.append(self.eos_token_id)
                    skip_tokens.append(self.split_lemma_token_id)
                    skip_tokens.append(self.nps_token_id)
                    skip_tokens.append(self.split_orth_token_id)
                    self.after_nps_token_tokens_to_skip = torch.Tensor(skip_tokens).int().to(scores.device)
                batch_scores[self.after_nps_token_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            elif last_token == self.split_elements_to_tag_id:
                if self.after_split_elements_to_tag_id_tokens_to_skip is None:
                    skip_tokens = deepcopy(self.tag_token_ids)
                    skip_tokens.append(self.split_elements_to_tag_id)
                    skip_tokens.append(self.eos_token_id)
                    skip_tokens.append(self.split_lemma_token_id)
                    skip_tokens.append(self.nps_token_id)
                    skip_tokens.append(self.split_orth_token_id)
                    self.after_split_elements_to_tag_id_tokens_to_skip = torch.Tensor(skip_tokens).int().to(scores.device)
                batch_scores[self.after_split_elements_to_tag_id_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            elif last_token in self.tag_token_ids:
                if self.after_tag_tokens_to_skip is None:
                    taggin_part_skip_tokens = list(range(scores.size()[1]))
                    for tag_token_id in self.tag_token_ids:
                        taggin_part_skip_tokens.remove(tag_token_id)
                    taggin_part_skip_tokens.remove(self.split_elements_to_tag_id)
                    taggin_part_skip_tokens.remove(self.eos_token_id)
                    self.after_tag_tokens_to_skip = torch.Tensor(taggin_part_skip_tokens).int().to(scores.device)
                batch_scores[self.after_tag_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            elif self.pre_answer_seq[-1] == last_token and self.pre_answer_seq == batch_input_ids[-len(self.pre_answer_seq):].tolist():
                if self.after_first_token_tokens_to_skip is None:
                    skip_tokens = deepcopy(self.tag_token_ids)
                    skip_tokens.append(self.split_elements_to_tag_id)
                    skip_tokens.append(self.eos_token_id)
                    skip_tokens.append(self.split_lemma_token_id)
                    skip_tokens.append(self.nps_token_id)
                    skip_tokens.append(self.split_orth_token_id)
                    self.after_first_token_tokens_to_skip = torch.Tensor(skip_tokens).int().to(scores.device)
                batch_scores[self.after_first_token_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            elif torch.sum(batch_input_ids == self.split_orth_token_id) <= torch.sum(batch_input_ids == self.split_lemma_token_id):
                if self.after_orth_token_tokens_to_skip is None:
                    skip_tokens = deepcopy(self.tag_token_ids)
                    skip_tokens.append(self.split_elements_to_tag_id)
                    skip_tokens.append(self.eos_token_id)
                    skip_tokens.append(self.nps_token_id)
                    skip_tokens.append(self.split_lemma_token_id)
                    self.after_orth_token_tokens_to_skip = torch.Tensor(skip_tokens).int().to(scores.device)
                batch_scores[self.after_orth_token_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
            else:
                if self.after_lemma_tokens_to_skip is None:
                    skip_tokens = deepcopy(self.tag_token_ids)
                    skip_tokens.append(self.split_elements_to_tag_id)
                    skip_tokens.append(self.eos_token_id)
                    skip_tokens.append(self.nps_token_id)
                    skip_tokens.append(self.split_orth_token_id)
                    self.after_lemma_tokens_to_skip = torch.Tensor(skip_tokens).int().to(scores.device)
                batch_scores[self.after_lemma_tokens_to_skip] = -torch.inf
                new_scores.append(batch_scores)
        return torch.stack(new_scores).to(scores.device)


def change_to_bytes(text):
    return list(text.encode('utf-8'))


byte_decoder = {v: k for k, v in bytes_to_unicode().items()}


def tokens_str_to_bytes(token):
    token = [byte_decoder[c] for c in token]
    return token


def split_element(element_string):
    element_dict = dict()
    temp = element_string.split(split_orth_token)
    assert len(temp) == 2
    element_dict['orth'] = temp[0]
    temp = temp[1].split(split_lemma_token)
    assert len(temp) == 2
    if nps_token in temp[0]:
        assert temp[0][:len(nps_token)] == nps_token
        temp[0] = temp[0][len(nps_token):]
        element_dict['nps'] = True
    else:
        element_dict['nps'] = False
    element_dict['lemma'] = temp[0].lower()
    element_dict['tagset'] = get_tagset(temp[1])
    return element_dict


def reconstruct_string(split_elements):
    text = ''
    split_elements[0]['nps'] = True
    for split_element in split_elements:
        if not split_element['nps']:
            text += ' '
        text += split_element['orth']
    return text


def calculate_string_reconstruction_error(labels_text, predistions_text):
    return distance(labels_text, predistions_text) / len(labels_text)


def cut_subset_based_on_error(orginal_text, split_elements_predictions):
    minial_error = float('inf')
    best_index = 0
    for i in range(1, len(split_elements_predictions)+1, 1):
        pred_str = reconstruct_string(split_elements_predictions[:i])
        error = calculate_string_reconstruction_error(orginal_text, pred_str)
        if error < minial_error:
            minial_error = error
            best_index = i
    return split_elements_predictions[:best_index]


def process_example(tokenizer, model_answer, answer, orginal_text=None, subset_by_text=False, align_by_key='orth'):
    if model_answer[-len(tokenizer.eos_token):] == tokenizer.eos_token:
        model_answer = model_answer[:-len(tokenizer.eos_token)]
        reached_end = True
    else:
        reached_end = False
    exact_example_match = model_answer == answer

    split_elements = answer.split(split_elements_to_tag)
    split_elements = [split_element(s_e) for s_e in split_elements]

    split_elements_model = model_answer.split(split_elements_to_tag)
    if not reached_end:
        split_elements_model.pop()
    split_elements_model = [split_element(s_e) for s_e in split_elements_model]

    gold_string = reconstruct_string(split_elements)

    if subset_by_text:
        if orginal_text is None:
            text = gold_string
        else:
            text = orginal_text
        split_elements_model = cut_subset_based_on_error(text, split_elements_model)

    if split_elements_model:
        pred_string = reconstruct_string(split_elements_model)
    else:
        pred_string = ''

    aligned_gold, aligned_pred = align_lists_with_nones([s[align_by_key] for s in split_elements], [s[align_by_key] for s in split_elements_model])

    fp_from_wrong_eos = 0
    for orth in reversed(aligned_gold):
        if orth is None:
            fp_from_wrong_eos += 1
        else:
            break

    dist = calculate_string_reconstruction_error(text, pred_string)
    length = len(text)
    str_error = calculate_string_reconstruction_error(text, pred_string)

    gold_dist = calculate_string_reconstruction_error(text, gold_string)

    both_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    lemmas_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    tags_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    orth_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    nps_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    all_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    elements_count = len(aligned_gold)

    s_e_m_i = 0
    s_e_i = 0

    assert [s[align_by_key] for s in split_elements] == [s for s in aligned_gold if s is not None]
    assert [s[align_by_key] for s in split_elements_model] == [s for s in aligned_pred if s is not None]

    for gold_lemma, pred_lemma in zip(aligned_gold, aligned_pred):
        assert gold_lemma is not None or pred_lemma is not None
        if gold_lemma is not None and pred_lemma is not None:
            s_e = split_elements[s_e_i]
            s_e_m = split_elements_model[s_e_m_i]
            if s_e['lemma'] == s_e_m['lemma']:
                lemmas_correct['tp'] += 1
                if s_e['tagset'] == s_e_m['tagset']:
                    tags_correct['tp'] += 1
                    both_correct['tp'] += 1
                    if s_e['orth'] == s_e_m['orth'] and s_e['nps'] == s_e_m['nps']:
                        all_correct['tp'] += 1
                    else:
                        all_correct['fp'] += 1
                        all_correct['fn'] += 1
                else:
                    all_correct['fp'] += 1
                    all_correct['fn'] += 1
                    tags_correct['fp'] += 1
                    tags_correct['fn'] += 1
                    both_correct['fp'] += 1
                    both_correct['fn'] += 1
            elif s_e['tagset'] == s_e_m['tagset']:
                tags_correct['tp'] += 1
                all_correct['fp'] += 1
                all_correct['fn'] += 1
                lemmas_correct['fp'] += 1
                lemmas_correct['fn'] += 1
                both_correct['fp'] += 1
                both_correct['fn'] += 1
            else:
                all_correct['fp'] += 1
                all_correct['fn'] += 1
                lemmas_correct['fp'] += 1
                lemmas_correct['fn'] += 1
                tags_correct['fp'] += 1
                tags_correct['fn'] += 1
                both_correct['fp'] += 1
                both_correct['fn'] += 1
            if s_e['orth'] == s_e_m['orth']:
                orth_correct['tp'] += 1
            else:
                orth_correct['fp'] += 1
                orth_correct['fn'] += 1
            if s_e['nps'] == s_e_m['nps']:
                nps_correct['tp'] += 1
            else:
                nps_correct['fp'] += 1
                nps_correct['fn'] += 1
        if gold_lemma is not None:
            s_e_i += 1
        if pred_lemma is not None:
            s_e_m_i += 1
        if gold_lemma is None and pred_lemma is not None:
            all_correct['fp'] += 1
            lemmas_correct['fp'] += 1
            tags_correct['fp'] += 1
            orth_correct['fp'] += 1
            nps_correct['fp'] += 1
            both_correct['fp'] += 1
        if gold_lemma is not None and pred_lemma is None:
            all_correct['fn'] += 1
            lemmas_correct['fn'] += 1
            tags_correct['fn'] += 1
            orth_correct['fn'] += 1
            nps_correct['fn'] += 1
            both_correct['fn'] += 1

    return int(exact_example_match), lemmas_correct, tags_correct, both_correct, orth_correct, nps_correct, all_correct, elements_count, dist, length, gold_dist, str_error


def nkjp_evaluate(accelerator, model, tokenizer, context_length, dataset_name, ds_dict_keys, batch_size, output_path, pickle_prefix=''):
    if isinstance(ds_dict_keys, list):
        ds_dict = datasets.load_dataset(dataset_name)
        test_dataset = concatenate_datasets([ds_dict[k] for k in ds_dict_keys])
    else:
        test_dataset = datasets.load_dataset(dataset_name)[ds_dict_keys]
    split_elements_to_tag_id = tokenizer.convert_tokens_to_ids(split_elements_to_tag)
    split_lemma_token_id = tokenizer.convert_tokens_to_ids(split_lemma_token)
    nps_token_id = tokenizer.convert_tokens_to_ids(nps_token)
    split_orth_token_id = tokenizer.convert_tokens_to_ids(split_orth_token)
    tag_token_ids = []
    for t_t in tag_special_tokens:
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

    force_tokens_logits_processor = ForceOrderLogitsProcessor(
        split_elements_to_tag_id, split_lemma_token_id, tag_token_ids, eos_token_id, split_orth_token_id, nps_token_id, pre_answer_seq
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
    batch = {'m': [], 'orginal_text': [], 'answer': []}
    for x in tqdm(test_dataset):
        j += 1
        if j <= len(result_list):
            continue
        m = x['messages']
        orginal_text = m[0]['content']
        orginal_text = re.sub('\s+', ' ', orginal_text)
        answer = m.pop()['content']
        batch['m'].append(m)
        batch['orginal_text'].append(orginal_text)
        batch['answer'].append(answer)

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

        batch = {'m': [], 'orginal_text': [], 'answer': []}

    distributed_state = PartialState()
    model.to(distributed_state.device)
    with distributed_state.split_between_processes(batches) as sub_batches:
        for batch in tqdm(sub_batches):
            max_new_tokens = batch['max_new_tokens']
            tokenized_prompt = batch['tokenized_prompt']
            prompt_lens = batch['prompt_lens']
            prompt = batch['prompt']
            tokenized_prompt = tokenized_prompt.to(distributed_state.device)
            with torch.no_grad():
                output = model.generate(tokenized_prompt, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, logits_processor=logits_processor)
            model_answer = tokenizer.batch_decode(output)

            model_answer = [m_a.replace(tokenizer.pad_token, '', -1) for m_a in model_answer]
            model_answer = [m_a[p_l:] for m_a, p_l in zip(model_answer, prompt_lens)]

            for m_a, a, o_t, p in zip(model_answer, batch['answer'], batch['orginal_text'], prompt):
                exact_example_match, lemmas_correct, tags_correct, both_correct, orth_correct, nps_correct, all_correct, elements_count, dist, length, gold_dist, str_error = process_example(tokenizer, m_a, a, align_by_key='orth', orginal_text=o_t, subset_by_text=True)
                result_list.append({
                    'answer': a,
                    'prompt': p,
                    'model_answer': m_a,
                    'exact_example_match': exact_example_match,
                    'lemmas_correct': lemmas_correct,
                    'tags_correct': tags_correct,
                    'both_correct': both_correct,
                    'orth_correct': orth_correct,
                    'nps_correct': nps_correct,
                    'all_correct': all_correct,
                    'elements_count': elements_count,
                    'dist': dist,
                    'length': length,
                    'gold_dist': gold_dist,
                    'str_error': str_error
                })

    result_list = gather_object(result_list)
    if result_list is not None and result_list:
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(result_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    exact_example_match = 0
    lemmas_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    tags_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    both_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    orth_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    nps_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    all_correct = {'tp': 0, 'fn': 0, 'fp': 0}
    elements_count = 0
    dist = 0
    length = 0
    str_error = 0
    gold_dist = 0

    for x in result_list:
        exact_example_match += x['exact_example_match']
        lemmas_correct = add_all_keys(lemmas_correct, x['lemmas_correct'])
        tags_correct = add_all_keys(tags_correct, x['tags_correct'])
        both_correct = add_all_keys(both_correct, x['both_correct'])
        orth_correct = add_all_keys(orth_correct, x['orth_correct'])
        nps_correct = add_all_keys(nps_correct, x['nps_correct'])
        all_correct = add_all_keys(all_correct, x['all_correct'])
        elements_count += x['elements_count']
        dist += x['dist']
        length += x['length']
        str_error += x['str_error']
        gold_dist += x['gold_dist']

    exact_example_match_score = exact_example_match * 100 / len(result_list)
    lemmas_correct_score = lemmas_correct['tp'] * 100 / (lemmas_correct['tp'] + 0.5 * (lemmas_correct['fp'] + lemmas_correct['fn']))
    tags_correct_score = tags_correct['tp'] * 100 / (tags_correct['tp'] + 0.5 * (tags_correct['fp'] + tags_correct['fn']))
    both_correct_score = both_correct['tp'] * 100 / (both_correct['tp'] + 0.5 * (both_correct['fp'] + both_correct['fn']))
    orth_correct_score = orth_correct['tp'] * 100 / (orth_correct['tp'] + 0.5 * (orth_correct['fp'] + orth_correct['fn']))
    nps_correct_score = nps_correct['tp'] * 100 / (nps_correct['tp'] + 0.5 * (nps_correct['fp'] + nps_correct['fn']))
    all_correct_score = all_correct['tp'] * 100 / (all_correct['tp'] + 0.5 * (all_correct['fp'] + all_correct['fn']))
    str_error_score = str_error / len(result_list)
    total_error_dist = dist / length

    print('test exact_example_match', exact_example_match_score)
    print('test lemmas f1', lemmas_correct_score)
    print('test tags f1', tags_correct_score)
    print('test both f1', both_correct_score)
    print('test orth f1', orth_correct_score)
    print('test nps f1', nps_correct_score)
    print('test all f1', all_correct_score)
    print('test elements count', elements_count)
    print('test examples from datasaet count', len(result_list))
    print('test mean of levenstein dist divided by gold str length.', str_error_score)
    print('test edit_dist per char of the input str', total_error_dist)

    score_dict = {
        'exact_example_match_score': exact_example_match_score,
        'lemmas_correct_score': lemmas_correct_score,
        'tags_correct_score': tags_correct_score,
        'both_correct_score': both_correct_score,
        'orth_correct_score': orth_correct_score,
        'nps_correct_score': nps_correct_score,
        'all_correct_score': all_correct_score,
        'elements_count': elements_count,
        'examples_in_ds': len(result_list),
        'mean_of_levenstein_dist_divided_by_gold_str_length': str_error_score,
        'edit_dist_per_char_of_the_input_str': total_error_dist
    }
    return score_dict
