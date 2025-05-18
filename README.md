# Inflectional-Tokenization

Polish lemma dictionary source:
http://download.sgjp.pl/morfeusz/20250511/polimorf-20250511.tab.gz

Czech lemma dictionary source:
https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3185

Fineweb2:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

Folder datasets contains NKJP, PDT and PolEval datasets transformed into a schema for generative model training.

The code uses the following repository code for cross-document attention masking (and as such pretrainig code reqiures the pretraning dataset to be compliant with it):
https://github.com/MeetKai/functionary/blob/main/functionary/train/packing/README.md

Open PL LLM Leaderboard:
https://huggingface.co/spaces/speakleash/open_pl_llm_leaderboard

The custom tokenizers use custom code so they can not be pickled to be saved, and as such, they will not save with the standard save_pretrained function from the transformers library. To save and load the custom tokenizers, appropriate functions are available at create_tokenizers/custom_tokenizer.py. The provided code handles them properly, given the appropriate custom 
custom config arguments.

The pretraining code assumes the provided dataset is already tokenized and packed to selected context length. It also assumes the attention mask filed to be following the format required by the cross-document attention masking repository.

The total batch size used in pretraining the models from the paper was 512 (with gradient accumulation set to 1).
The supervised finetuning total batch size for instruction tuning, NKJP dataset, and PDT dataset used in finetuning the models from the paper was 128 (with gradient accumulation set to 1).