from datasets import load_from_disk, DatasetDict


from create_tokenizer.custom_tokenizer import create_tokenizer, get_trainer, save_tokenizer



def batch_iterator(dataset, batch_size):
    tok_dataset = dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


def train(
        dataest_path='polish_subset_of_fineweb2',
        save_path='new_tokernizer/tokenizer.json',
        vocab_size=40960,
        special_tokens=None,
        use_bos=False,
        use_eos=True,
        lang='pl_lcs',
        dictionary_path='split_dictionary_pl_lcs.pkl',
        ready_splits_path=None
):
    batch_size = 10_000
    dataset = load_from_disk(dataest_path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset['train']
    tokenizer = create_tokenizer(lang=lang, use_bos=use_bos, use_eos=use_eos, dictionary_path=dictionary_path, ready_splits_path=ready_splits_path)
    trainer = get_trainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(batch_iterator(dataset=dataset, batch_size=batch_size), trainer=trainer, length=len(dataset))
    save_tokenizer(tokenizer=tokenizer, file=save_path, lang=lang)
    return


if __name__ == '__main__':
    train()
