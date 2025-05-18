from typing import List
import pickle
from pathlib import Path


from tokenizers import NormalizedString, PreTokenizedString
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer, Regex
from transformers import PreTrainedTokenizerFast
from tokenizers import processors


bos_token = "<|begin_of_text|>"
eos_token = "<|end_of_text|>"


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class SuffixMatcher:
    def __init__(self, suffixes):
        self.root = TrieNode()
        for suffix in suffixes:
            self._insert(suffix[::-1])

    def _insert(self, reversed_suffix):
        node = self.root
        for char in reversed_suffix:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def longest_suffix(self, word):
        node = self.root
        longest = ''
        current = ''
        for char in reversed(word):
            if char in node.children:
                node = node.children[char]
                current += char
                if node.is_end:
                    longest = current
            else:
                break
        return longest[::-1]


class DesinanceSplit:
    def __init__(self, split_dictionary, ready_splits=None, lang='pl'):
        self.split_dictionary = split_dictionary[0]
        self.suffixes = split_dictionary[1]
        self.matcher = SuffixMatcher(self.suffixes)
        self.ready_splits = ready_splits
        assert lang in ['pl', 'cz']
        if lang == 'pl':
            self.alphabet = set("AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSTUVWXYZŹŻaąbcćdeęfghijklłmnńoópqrstvuwxyzźż")
        else:
            self.alphabet = set("AÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽaábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž")

    def does_not_contain_any(self, s):
        return set(s).isdisjoint(set(self.alphabet))

    def format_dict_key(self, word):
        if not word:
            return word
        if len(word) > 1:
            word = word[0].lower() + word[1:]
        else:
            word = word.lower()
        return word

    def longest_suffix(self, word):
        return self.matcher.longest_suffix(word)

    def get_normalized_split_str(self, key, normalized_string, raw_string, striped_string):
        split_idxs = self.split_dictionary.get(key, None)
        if split_idxs is None:
            return None
        if len(split_idxs) == 1:
            return [normalized_string]
        splited_normalized_str = []
        offset = raw_string.find(striped_string[0])
        split_idxs = [
            [t[0] + offset, t[1] + offset]
            for t in split_idxs
        ]
        split_idxs[0][0] -= offset
        split_idxs[-1][-1] = len(raw_string)
        splited_normalized_str = [normalized_string.slice((pair[0], pair[1])) for pair in split_idxs]
        return splited_normalized_str

    def _dictionary_split(self, normalized_string: NormalizedString) -> List[NormalizedString]:
        raw_string = str(normalized_string)
        if self.ready_splits is not None:
            splits = self.ready_splits.get(raw_string, None)
            if splits is not None and splits != (None, None):
                return [normalized_string.slice(pair) for pair in splits]
        striped_string = raw_string.strip()
        if not striped_string:
            return [normalized_string]
        key = self.format_dict_key(striped_string)
        splited_normalized_str = self.get_normalized_split_str(key, normalized_string, raw_string, striped_string)
        if splited_normalized_str is None:
            if self.does_not_contain_any(striped_string):
                return [normalized_string]
            ls = self.longest_suffix(key)
            if ls:
                end_offset = len(raw_string) - raw_string.rfind(striped_string[-1]) - 1
                output_list = [normalized_string.slice((0, len(raw_string)-len(ls)-end_offset)), normalized_string.slice((len(raw_string)-len(ls)-end_offset, len(raw_string)))]
                return output_list
            else:
                return [normalized_string]
        else:
            return splited_normalized_str

    def dictionary_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        return self._dictionary_split(normalized_string)

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.dictionary_split)


class DesinanceSplitSuffix:
    def __init__(self, split_dictionary):
        self.split_dictionary = split_dictionary

    def format_dict_key(self, word):
        if not word:
            return word
        word = word[0].lower() + word[1:]
        return word

    def dictionary_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        raw_string = str(normalized_string)
        striped_string = raw_string.strip()
        key = self.format_dict_key(striped_string)
        if key in self.split_dictionary:
            idx = self.split_dictionary[key][0]
            pre_idx = raw_string.find(striped_string)
            total_idx = idx + pre_idx
            if len(raw_string) == total_idx:
                return [normalized_string]
            else:
                return [normalized_string[:total_idx], normalized_string[total_idx:]]
        else:
            return [normalized_string]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.dictionary_split)


def get_pretokenizer(lang='pl', dictionary_path=None, ready_splits_path=None):
    pre_split = pre_tokenizers.Split(
        pattern=Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
        behavior="isolated"
    )
    pre_bytes = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    if lang:
        if lang == 'pl_lcs':
            if dictionary_path is None:
                dictionary_path = 'split_dictionary_pl_lcs.pkl'
            with open(dictionary_path, 'rb') as f:
                split_dictionary = pickle.load(f)
            if ready_splits_path is not None:
                with open(ready_splits_path, 'rb') as f:
                    ready_splits = pickle.load(f)
            else:
                ready_splits = None
            pre_desi = pre_tokenizers.PreTokenizer.custom(DesinanceSplit(split_dictionary=split_dictionary, ready_splits=ready_splits, lang='pl'))
        elif lang == 'pl_suffix':
            if dictionary_path is None:
                dictionary_path = 'split_dictionary_pl_suffix.pkl'
            with open(dictionary_path, 'rb') as f:
                split_dictionary = pickle.load(f)
            pre_desi = pre_tokenizers.PreTokenizer.custom(DesinanceSplitSuffix(split_dictionary=split_dictionary))
        elif lang == 'cz_lcs':
            if dictionary_path is None:
                dictionary_path = 'split_dictionary_czech_lcs.pkl'
            with open(dictionary_path, 'rb') as f:
                split_dictionary = pickle.load(f)
            if ready_splits_path is not None:
                with open(ready_splits_path, 'rb') as f:
                    ready_splits = pickle.load(f)
            else:
                ready_splits = None
            pre_desi = pre_tokenizers.PreTokenizer.custom(DesinanceSplit(split_dictionary=split_dictionary, ready_splits=ready_splits, lang='cz'))
        else:
            assert False
        return pre_tokenizers.Sequence([pre_split, pre_desi, pre_bytes])
    else:
        return pre_tokenizers.Sequence([pre_split, pre_bytes])


def create_tokenizer(lang='pl', dictionary_path=None, use_bos=False, use_eos=True, ready_splits_path=None):
    tokenizer = Tokenizer(models.BPE(dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="", fuse_unk=False, byte_fallback=False, ignore_merges=True))
    tokenizer.version = "1.0"
    tokenizer.pre_tokenizer = get_pretokenizer(desinance=lang, dictionary_path=dictionary_path, ready_splits_path=ready_splits_path)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)
    pp_byte = processors.ByteLevel(add_prefix_space=True, trim_offsets=False, use_regex=True)
    if use_bos:
        if use_eos:
            single = f"{bos_token}:0 $A:0 {eos_token}:0"
            pair = f"{bos_token}:0 $A:0 {eos_token}:0 {bos_token}:1 $B:1 {eos_token}:1"
        else:
            single = f"{bos_token}:0 $A:0"
            pair = f"{bos_token}:0 $A:0 {bos_token}:1 $B:1"
    else:
        if use_eos:
            single = f"$A:0 {eos_token}:0"
            pair = f"$A:0 {eos_token}:0 $B:1 {eos_token}:1"
        else:
            single = "$A:0"
            pair = "$A:0 $B:1"
    pp_tp = processors.TemplateProcessing(
        single=single,
        pair=pair,
        special_tokens=[(bos_token, 0), (eos_token, 1)]
        )
    tokenizer.post_processor = processors.Sequence([pp_byte, pp_tp])
    return tokenizer


def save_tokenizer(tokenizer, file, lang=True):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if lang:
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(file))
    else:
        tokenizer.save(str(file))
    return


def load_tokenizer(file, fast=True, tokenizer_kwargs=None, lang=True, dictionary_path=None, ready_splits_path=None):
    tokenizer = Tokenizer.from_file(file)
    if fast:
        if tokenizer_kwargs is not None:
            fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **tokenizer_kwargs)
        else:
            fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        if lang is not None and lang:
            fast_tokenizer._tokenizer.pre_tokenizer = get_pretokenizer(lang=lang, dictionary_path=dictionary_path, ready_splits_path=ready_splits_path)
        return fast_tokenizer
    else:
        if lang is not None and lang:
            tokenizer.pre_tokenizer = get_pretokenizer(lang=lang, dictionary_path=dictionary_path, ready_splits_path=ready_splits_path)
        return tokenizer


def save_tokenize_fast(tokenizer, file, lang=True):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if lang:
        tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save_pretrained(file)
    return


def get_trainer(vocab_size=32000, special_tokens=None):
    if special_tokens is None:
        special_tokens=[bos_token, eos_token]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=special_tokens, show_progress=True)
    return trainer
