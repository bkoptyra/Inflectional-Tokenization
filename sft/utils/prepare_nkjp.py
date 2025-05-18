import torch

split_elements_to_tag = '<|split_elements_to_tag|>'
split_lemma_token = '<|split_lemma_from_tags|>'
nps_token = '<|no_space_token|>'
split_orth_token = '<|split_orth_from_lemma|>'


new_special_tokens = [split_elements_to_tag, split_lemma_token, split_orth_token]

tag_special_tokens = [
    '<|special_tag_token_adv|>',
    '<|special_tag_token_com|>',
    '<|special_tag_token_bedzie|>',
    '<|special_tag_token_num|>',
    '<|special_tag_token_adj|>',
    '<|special_tag_token_perf|>',
    '<|special_tag_token_agl|>',
    '<|special_tag_token_inst|>',
    '<|special_tag_token_m3|>',
    '<|special_tag_token_akc|>',
    '<|special_tag_token_ter|>',
    '<|special_tag_token_aglt|>',
    '<|special_tag_token_impt|>',
    '<|special_tag_token_winien|>',
    '<|special_tag_token_loc|>',
    '<|special_tag_token_qub|>',
    '<|special_tag_token_xxx|>',
    '<|special_tag_token_pant|>',
    '<|special_tag_token_n|>',
    '<|special_tag_token_nom|>',
    '<|special_tag_token_siebie|>',
    '<|special_tag_token_nakc|>',
    '<|special_tag_token_sec|>',
    '<|special_tag_token_nagl|>',
    '<|special_tag_token_burk|>',
    '<|special_tag_token_imperf|>',
    '<|special_tag_token_imps|>',
    '<|special_tag_token_comp|>',
    '<|special_tag_token_dat|>',
    '<|special_tag_token_adjp|>',
    '<|special_tag_token_pun|>',
    '<|special_tag_token_subst|>',
    '<|special_tag_token_voc|>',
    '<|special_tag_token_gen|>',
    '<|special_tag_token_neg|>',
    '<|special_tag_token_pcon|>',
    '<|special_tag_token_congr|>',
    '<|special_tag_token_prep|>',
    '<|special_tag_token_inf|>',
    '<|special_tag_token_pos|>',
    '<|special_tag_token_praep|>',
    '<|special_tag_token_nwok|>',
    '<|special_tag_token_brev|>',
    '<|special_tag_token_interp|>',
    '<|special_tag_token_pl|>',
    '<|special_tag_token_ppron3|>',
    '<|special_tag_token_npraep|>',
    '<|special_tag_token_numcol|>',
    '<|special_tag_token_f|>',
    '<|special_tag_token_m1|>',
    '<|special_tag_token_pri|>',
    '<|special_tag_token_ppron12|>',
    '<|special_tag_token_aff|>',
    '<|special_tag_token_praet|>',
    '<|special_tag_token_m2|>',
    '<|special_tag_token_interj|>',
    '<|special_tag_token_conj|>',
    '<|special_tag_token_adja|>',
    '<|special_tag_token_fin|>',
    '<|special_tag_token_depr|>',
    '<|special_tag_token_sup|>',
    '<|special_tag_token_ger|>',
    '<|special_tag_token_acc|>',
    '<|special_tag_token_pred|>',
    '<|special_tag_token_adjc|>',
    '<|special_tag_token_wok|>',
    '<|special_tag_token_rec|>',
    '<|special_tag_token_pact|>',
    '<|special_tag_token_sg|>',
    '<|special_tag_token_npun|>',
    '<|special_tag_token_ppas|>'
]


tag_penalties_dict = {}
for tag in tag_special_tokens:
    tag_penalties_dict[tag] = 1


def prepare_penalizing_nkjp_tags(tokenizer):
    tag_token_id_to_penalty = {}
    tag_penalties_dict[tag]
    for tag_token in tag_special_tokens:
        tag_token_id_to_penalty[tokenizer.convert_tokens_to_ids(tag_token)] = tag_penalties_dict[tag_token]

    def penalize_nkjp_tags(loss, target):
        for tag_id, penalty in tag_token_id_to_penalty.items():
            tag_mask = (target == tag_id)
            loss += penalty * (tag_mask.float() * loss)
        return loss

    return penalize_nkjp_tags


def prepare_tokenizer(tokenizer):
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    tokenizer.add_tokens(tag_special_tokens, special_tokens=True)
    tokenizer.add_tokens([nps_token], special_tokens=True)
    return tokenizer


def create_compute_metrics_nkjp(tokenizer):

    tag_special_tokens_tokenizer = []
    tokenizer_sep_tokens = []
    for split_token in new_special_tokens:
        tokenizer_sep_tokens.append(tokenizer.convert_tokens_to_ids(split_token))
    for tag_token in tag_special_tokens:
        tag_special_tokens_tokenizer.append(tokenizer.convert_tokens_to_ids(tag_token))

    eos_token_id = tokenizer.eos_token_id

    def compute_metrics_nkjp(eval_preds):
        preds, labels = eval_preds
        preds = torch.Tensor(preds).int()
        labels = torch.Tensor(labels[..., 1:]).contiguous().int()
        mask = labels != -100
        correct_predictions = (preds == labels) & mask
        total_tokens = mask.sum()
        correct_tokens = correct_predictions.sum()
        accuracy = (correct_tokens.sum() / total_tokens.sum()).item() \
            if total_tokens.sum() > 0 else 0.0

        only_tags_mask = torch.isin(labels, torch.tensor(tag_special_tokens_tokenizer))
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

    return compute_metrics_nkjp
