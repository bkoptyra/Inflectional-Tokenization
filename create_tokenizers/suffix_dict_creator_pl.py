import csv
import pickle
import os

import pandas as pd


def utf8len(s):
    return len(s.encode('utf-8'))


def get_longest_common_prefix(word_list):
    return os.path.commonprefix(word_list)


def format_dict_key(word):
    word = word[0].lower() + word[1:]
    return word


def main():
    with open('polimorf-20240609.tab', 'r') as rows_file:
        characters = 0
        endline_len = utf8len('\n')
        while True:
            line = rows_file.readline()
            characters += utf8len(line) + endline_len
            if line.strip() == '#</COPYRIGHT>':
                break
        rows_file.seek(characters - 1)                                            
        rows_reader = csv.reader(rows_file, delimiter='\t')
        all_rows = []
        for row in rows_reader:
            all_rows.append(row)
    df = pd.DataFrame(all_rows, columns=['word', 'lemma', 'pos_tags', '?', '??'])
    df[['pos', 'plurality', 'other_pos_tags']] = df['pos_tags'].str.split(':', n=2, expand=True)
    meaning_groups = df.groupby(['lemma', 'pos', 'plurality'])

    split_dictionary = {}
    for group_key, group in meaning_groups:
        prefix = get_longest_common_prefix([format_dict_key(word) for word in list(group['word'].to_dict().values())])
        for _, row in group.iterrows():
            word = row['word']
            word = format_dict_key(word)
            suffix = word[len(prefix):]
            if (word in split_dictionary) and not (split_dictionary[word] == (prefix, suffix)):
                if split_dictionary[word][0] > len(prefix):
                    assert(split_dictionary[word][1][0][:len(prefix)] == prefix)
                    split_dictionary[word] = (len(prefix), (prefix, suffix))
                else:
                    assert(prefix[:split_dictionary[word][0]] == split_dictionary[word][1][0])
            else:
                split_dictionary[word] = (len(prefix), (prefix, suffix))

    with open('split_dictionary_pl_suffix.pkl', 'wb') as f:
        pickle.dump(split_dictionary, f)


if __name__ == '__main__':
    main()
