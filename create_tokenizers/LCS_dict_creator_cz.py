import csv
import pickle
import os
import difflib

import pandas as pd


def utf8len(s):
    return len(s.encode('utf-8'))


def get_longest_common_prefix(word_list):
    return os.path.commonprefix(word_list)


def format_dict_key(word):
    if len(word) > 1:
        word = word[0].lower() + word[1:]
    else:
        word = word.lower()
    return word


def longest_common_subsequence(a: str, b: str) -> str:
    matcher = difflib.SequenceMatcher(None, a, b)
    matches = [a[i:i + n] for i, j, n in matcher.get_matching_blocks() if n > 0]
    return max(matches, key=len, default='')


def find_substring_indices(string: str, substring: str):
    start = string.find(substring)
    if start == -1:
        return [0, 0]
    end = start + len(substring)
    return [start, end]


def main():
    file_path = 'czech-morfflex-2.1.tsv.xz'
    df = pd.read_csv(file_path, sep='\t', compression='xz')
    df = df.fillna('')
    meaning_groups = df.groupby(['§'])

    split_dictionary = {}
    for group_key, group in meaning_groups:
        words = list(group['§.1'].to_dict().values())
        words = [format_dict_key(word) for word in words]
        lemma = group_key[0]
        if '_' in lemma:
            lemma = lemma.split('_')[0]
            lemma = lemma.split('-')[0]
        lemma = format_dict_key(lemma)
        longest_sequance = lemma
        for word in words:
            lcs = longest_common_subsequence(lemma, word)
            if len(lcs) < len(longest_sequance):
                longest_sequance = lcs
        for _, row in group.iterrows():
            word = row['§.1']
            word = format_dict_key(word)
            lcs = longest_common_subsequence(lemma, word)
            split_idxs = []
            split_idxs += find_substring_indices(word, longest_sequance)
            split_idxs += find_substring_indices(word, lcs)
            split_idxs = [si for si in split_idxs if (si != 0 and si != len(word))]
            split_idxs = sorted(list(set(split_idxs)))
            split_word = []
            split_word_idxs = []
            prev_idx = 0
            for index in split_idxs:
                split_word_idxs.append(index)
                split_word.append(word[prev_idx:index])
                prev_idx = index
            split_word.append(word[prev_idx:len(word)])
            if (word in split_dictionary) and not (split_dictionary[word] == (split_word_idxs, split_word)):
                if len(split_dictionary[word][0]) < len(split_word_idxs):
                    split_dictionary[word] = (split_word_idxs, split_word)
            else:
                split_dictionary[word] = (split_word_idxs, split_word)

    with open('split_dictionary_czech_lcs.pkl', 'wb') as f:
        pickle.dump(split_dictionary, f)


if __name__ == '__main__':
    main()
