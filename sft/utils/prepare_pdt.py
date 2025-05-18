import torch


pdt_split_elements_to_tag = '<|split_elements_to_tag|>'
pdt_split_lemma_token = '<|split_lemma_from_tags|>'
pdt_split_orth_token = '<|split_orth_from_lemma|>'

pdt_tag_special_tokens = [
    '<|special_CASE_token_1|>',
    '<|special_CASE_token_2|>',
    '<|special_CASE_token_3|>',
    '<|special_CASE_token_4|>',
    '<|special_CASE_token_5|>',
    '<|special_CASE_token_6|>',
    '<|special_CASE_token_7|>',
    '<|special_CASE_token_X|>',
    '<|special_GENDER_token_F|>',
    '<|special_GENDER_token_H|>',
    '<|special_GENDER_token_I|>',
    '<|special_GENDER_token_M|>',
    '<|special_GENDER_token_N|>',
    '<|special_GENDER_token_Q|>',
    '<|special_GENDER_token_T|>',
    '<|special_GENDER_token_X|>',
    '<|special_GENDER_token_Y|>',
    '<|special_GENDER_token_Z|>',
    '<|special_GRADE_token_1|>',
    '<|special_GRADE_token_2|>',
    '<|special_GRADE_token_3|>',
    '<|special_NEGATION_token_A|>',
    '<|special_NEGATION_token_N|>',
    '<|special_NUMBER_token_D|>',
    '<|special_NUMBER_token_P|>',
    '<|special_NUMBER_token_S|>',
    '<|special_NUMBER_token_W|>',
    '<|special_NUMBER_token_X|>',
    '<|special_PERSON_token_1|>',
    '<|special_PERSON_token_2|>',
    '<|special_PERSON_token_3|>',
    '<|special_POSSGENDER_token_F|>',
    '<|special_POSSGENDER_token_M|>',
    '<|special_POSSGENDER_token_X|>',
    '<|special_POSSGENDER_token_Z|>',
    '<|special_POSSNUMBER_token_P|>',
    '<|special_POSSNUMBER_token_S|>',
    '<|special_POS_token_A|>',
    '<|special_POS_token_B|>',
    '<|special_POS_token_C|>',
    '<|special_POS_token_D|>',
    '<|special_POS_token_F|>',
    '<|special_POS_token_I|>',
    '<|special_POS_token_J|>',
    '<|special_POS_token_N|>',
    '<|special_POS_token_P|>',
    '<|special_POS_token_Q|>',
    '<|special_POS_token_R|>',
    '<|special_POS_token_S|>',
    '<|special_POS_token_T|>',
    '<|special_POS_token_V|>',
    '<|special_POS_token_Z|>',
    '<|special_RESERVE1_token_B|>',
    '<|special_RESERVE1_token_I|>',
    '<|special_RESERVE1_token_P|>',
    '<|special_RESERVE2_token_c|>',
    '<|special_RESERVE2_token_e|>',
    '<|special_RESERVE2_token_m|>',
    '<|special_RESERVE2_token_o|>',
    '<|special_RESERVE2_token_s|>',
    '<|special_SUBPOS_token_%|>',
    '<|special_SUBPOS_token_*|>',
    '<|special_SUBPOS_token_,|>',
    '<|special_SUBPOS_token_1|>',
    '<|special_SUBPOS_token_2|>',
    '<|special_SUBPOS_token_3|>',
    '<|special_SUBPOS_token_4|>',
    '<|special_SUBPOS_token_5|>',
    '<|special_SUBPOS_token_6|>',
    '<|special_SUBPOS_token_7|>',
    '<|special_SUBPOS_token_8|>',
    '<|special_SUBPOS_token_9|>',
    '<|special_SUBPOS_token_:|>',
    '<|special_SUBPOS_token_=|>',
    '<|special_SUBPOS_token_A|>',
    '<|special_SUBPOS_token_B|>',
    '<|special_SUBPOS_token_C|>',
    '<|special_SUBPOS_token_D|>',
    '<|special_SUBPOS_token_E|>',
    '<|special_SUBPOS_token_F|>',
    '<|special_SUBPOS_token_G|>',
    '<|special_SUBPOS_token_H|>',
    '<|special_SUBPOS_token_I|>',
    '<|special_SUBPOS_token_K|>',
    '<|special_SUBPOS_token_L|>',
    '<|special_SUBPOS_token_M|>',
    '<|special_SUBPOS_token_N|>',
    '<|special_SUBPOS_token_O|>',
    '<|special_SUBPOS_token_P|>',
    '<|special_SUBPOS_token_Q|>',
    '<|special_SUBPOS_token_R|>',
    '<|special_SUBPOS_token_S|>',
    '<|special_SUBPOS_token_T|>',
    '<|special_SUBPOS_token_U|>',
    '<|special_SUBPOS_token_V|>',
    '<|special_SUBPOS_token_W|>',
    '<|special_SUBPOS_token_Y|>',
    '<|special_SUBPOS_token_Z|>',
    '<|special_SUBPOS_token_^|>',
    '<|special_SUBPOS_token_a|>',
    '<|special_SUBPOS_token_b|>',
    '<|special_SUBPOS_token_c|>',
    '<|special_SUBPOS_token_d|>',
    '<|special_SUBPOS_token_e|>',
    '<|special_SUBPOS_token_f|>',
    '<|special_SUBPOS_token_g|>',
    '<|special_SUBPOS_token_h|>',
    '<|special_SUBPOS_token_i|>',
    '<|special_SUBPOS_token_j|>',
    '<|special_SUBPOS_token_l|>',
    '<|special_SUBPOS_token_m|>',
    '<|special_SUBPOS_token_n|>',
    '<|special_SUBPOS_token_o|>',
    '<|special_SUBPOS_token_p|>',
    '<|special_SUBPOS_token_r|>',
    '<|special_SUBPOS_token_s|>',
    '<|special_SUBPOS_token_t|>',
    '<|special_SUBPOS_token_v|>',
    '<|special_SUBPOS_token_w|>',
    '<|special_SUBPOS_token_y|>',
    '<|special_SUBPOS_token_z|>',
    '<|special_SUBPOS_token_}|>',
    '<|special_TENSE_token_F|>',
    '<|special_TENSE_token_P|>',
    '<|special_TENSE_token_R|>',
    '<|special_TENSE_token_X|>',
    '<|special_VAR_token_1|>',
    '<|special_VAR_token_2|>',
    '<|special_VAR_token_3|>',
    '<|special_VAR_token_4|>',
    '<|special_VAR_token_5|>',
    '<|special_VAR_token_6|>',
    '<|special_VAR_token_7|>',
    '<|special_VAR_token_8|>',
    '<|special_VAR_token_9|>',
    '<|special_VAR_token_a|>',
    '<|special_VAR_token_b|>',
    '<|special_VAR_token_c|>',
    '<|special_VOICE_token_A|>',
    '<|special_VOICE_token_P|>'
]

pdt_new_special_tokens = [pdt_split_elements_to_tag, pdt_split_lemma_token, pdt_split_orth_token]

pdt_tag_penalties_dict = {}
for tag in pdt_tag_special_tokens:
    pdt_tag_penalties_dict[tag] = 1


def prepare_penalizing_pdt_tags(tokenizer):
    tag_token_id_to_penalty = {}
    pdt_tag_penalties_dict[tag]
    for tag_token in pdt_tag_special_tokens:
        tag_token_id_to_penalty[tokenizer.convert_tokens_to_ids(tag_token)] = pdt_tag_penalties_dict[tag_token]

    def penalize_pdt_tags(loss, target):
        for tag_id, penalty in tag_token_id_to_penalty.items():
            tag_mask = (target == tag_id)
            loss += penalty * (tag_mask.float() * loss)
        return loss

    return penalize_pdt_tags


def pdt_prepare_tokenizer(tokenizer):
    tokenizer.add_tokens(pdt_new_special_tokens, special_tokens=True)
    tokenizer.add_tokens(pdt_tag_special_tokens, special_tokens=True)
    return tokenizer


def create_compute_metrics_pdt(tokenizer):
    pdt_tag_special_tokens_tokenizer = []
    tokenizer_sep_tokens = []
    for split_token in pdt_new_special_tokens:
        tokenizer_sep_tokens.append(tokenizer.convert_tokens_to_ids(split_token))
    for tag_token in pdt_tag_special_tokens:
        pdt_tag_special_tokens_tokenizer.append(tokenizer.convert_tokens_to_ids(tag_token))

    eos_token_id = tokenizer.eos_token_id

    def compute_metrics_pdt(eval_preds):
        preds, labels = eval_preds
        preds = torch.Tensor(preds).int()
        labels = torch.Tensor(labels[..., 1:]).contiguous().int()
        mask = labels != -100
        correct_predictions = (preds == labels) & mask
        total_tokens = mask.sum()
        correct_tokens = correct_predictions.sum()
        accuracy = (correct_tokens.sum() / total_tokens.sum()).item() \
            if total_tokens.sum() > 0 else 0.0

        only_tags_mask = torch.isin(labels, torch.tensor(pdt_tag_special_tokens_tokenizer))
        tags_mask = mask & only_tags_mask

        only_eos_mask = labels == eos_token_id
        eos_mask = mask & only_eos_mask

        correct_eos_tokens_predictions = (preds == labels) & eos_mask
        total_eos_tokens = eos_mask.sum()
        eos_token_accuracy = (correct_eos_tokens_predictions.sum() / total_eos_tokens.sum()).item() \
            if total_eos_tokens.sum() > 0 else 0.0

        lemma_mask = ~torch.isin(labels, torch.tensor(tokenizer_sep_tokens))
        lemma_mask = lemma_mask & mask & ~only_tags_mask

        correct_lemma_tokens_predictions = (preds == labels) & lemma_mask
        total_lemma_tokens = lemma_mask.sum()
        lemma_tokens_accuracy = (correct_lemma_tokens_predictions.sum() / total_lemma_tokens.sum()).item() \
            if total_lemma_tokens.sum() > 0 else 0.0
        correct_tag_tokens_predictions = (preds == labels) & tags_mask
        total_tag_tokens = tags_mask.sum()
        tags_token_accuracy = (correct_tag_tokens_predictions.sum() / total_tag_tokens.sum()).item() \
            if total_tag_tokens.sum() > 0 else 0.0

        total_lemma_chunks = 0
        correct_lemma_chunks = 0
        for row_mask, row_pred in zip(lemma_mask, correct_lemma_tokens_predictions):
            still_correct = True
            in_chunk = False
            for element_mask, element_pred in zip(row_mask, row_pred):
                if element_mask:
                    in_chunk = True
                    if not element_pred:
                        still_correct = False
                else:
                    if in_chunk:
                        total_lemma_chunks += 1
                        if still_correct:
                            correct_lemma_chunks += 1
                        still_correct = True
                        in_chunk = False
        lemma_exact_accuracy = correct_lemma_chunks / total_lemma_chunks

        output = dict()
        output['everything_not_tag_exact_accuracy'] = lemma_exact_accuracy
        output['everything_not_tag_tokens_accuracy'] = lemma_tokens_accuracy
        output['tags_token_accuracy'] = tags_token_accuracy
        output['eos_token_accuracy'] = eos_token_accuracy
        output['mean_token_accuracy'] = accuracy
        return output

    return compute_metrics_pdt


def keep_last_true(tensor):
    last_true_idx = (tensor.nonzero(as_tuple=True)[0][-1]) if tensor.any() else None
    result = torch.zeros_like(tensor, dtype=torch.bool)
    if last_true_idx is not None:
        result[last_true_idx] = True
    return result


def mask_orth_and_eos_tokens(labels, tokenizer):
    split_pdt_orth_from_lemma_id = tokenizer.convert_tokens_to_ids(pdt_split_orth_token)
    split_pdt_split_elements_to_tag = tokenizer.convert_tokens_to_ids(pdt_split_elements_to_tag)
    pdt_eos_id = tokenizer.eos_token_id
    token_1_mask = (labels == split_pdt_orth_from_lemma_id)
    token_2_mask = (labels == split_pdt_split_elements_to_tag)
    eos_mask = torch.nonzero(labels == pdt_eos_id)
    orth_mask = torch.zeros_like(labels, dtype=torch.bool)
    token_1_indices = torch.nonzero(token_1_mask)
    token_2_indices = torch.nonzero(token_2_mask)
    orth_mask[:token_1_indices[0]] = True
    i, j = 1, 0
    while i < len(token_1_indices) and j < len(token_2_indices):
        if token_1_indices[i] > token_2_indices[j]:
            orth_mask[(token_2_indices[j] + 1):(token_1_indices[i] + 1)] = True
            j += 1
        assert i == j
        i += 1
    orth_mask[token_1_mask] = True
    orth_mask[eos_mask] = True
    labels[orth_mask] = -100
    return labels
