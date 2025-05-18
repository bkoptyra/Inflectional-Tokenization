from transformers import AutoTokenizer


from utils.custom_tokenizer import load_tokenizer


def get_tokenizer(model_args, custom_args):
    tokenizer_kwargs = {
        'cache_dir': model_args['cache_dir'],
        'use_fast': model_args['use_fast_tokenizer'],
        'revision': model_args['model_revision'],
        'trust_remote_code': model_args['trust_remote_code'],
    }

    if custom_args['custom_tokenizer']:
        tokenizer = load_tokenizer(
            file=model_args['tokenizer_name'],
            fast=True,
            lang=custom_args['lang'],
            dictionary_path=custom_args['dictionary_path']
        )
    elif 'tokenizer_name' in model_args and model_args['tokenizer_name']:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args['tokenizer_name'],
            **tokenizer_kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args['pretrained_model_name_or_path'],
            **tokenizer_kwargs
        )
    if (
        'chat_template' in custom_args
        and custom_args['chat_template'] is not None
        and custom_args['chat_template']
    ):
        tokenizer.chat_template = custom_args['chat_template']
        if custom_args['completion_only']:
            instruction_template = custom_args['instruction_template']
            response_template = custom_args['response_template']
            if custom_args['add_prefix_tokens']:
                special_tokens = [
                    custom_args['bop_token'],
                    custom_args['eop_token']
                ]
                tokenizer.add_tokens(special_tokens, special_tokens=True)
        print(f"Use chat template from input args: {tokenizer.chat_template}")
    else:
        if tokenizer.chat_template is not None:
            print(
                f"Use default tokenizer chat template: \
                    {tokenizer.chat_template}"
            )
            instruction_template = custom_args['instruction_template']
            response_template = custom_args['response_template']
        else:
            bop_token = "<|beggining_of_instruction_token_[INST]|>"
            eop_token = "<|end_of_instruction_token_[/INST]|>"
            eos_token = "<|end_of_sentence_token_[EOS]|>"
            instruction_template = bop_token + "user" + eop_token
            response_template = bop_token + "assistant" + eop_token
            system_template = bop_token + "system" + eop_token
            tokenizer.bop_token = bop_token
            tokenizer.eop_token = eop_token
            tokenizer.eos_token = eos_token

            def chat_template(self):
                self.assistant = response_template
                self.user = instruction_template
                self.system = system_template
                return (
                    "{% for message in messages %}"
                    f"{{{{'{self.bop_token}' + message['role'] + \
                        '{self.eop_token}' + '\n' + message['content'] + \
                            '{self.eos_token}' + '\n'}}}}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    f"{{{{ '{self.assistant}\n' }}}}"
                    "{% endif %}"
                )
            special_tokens = [bop_token, eop_token, eos_token]
            tokenizer.add_tokens(special_tokens, special_tokens=True)
            tokenizer.chat_template = chat_template(tokenizer)
            print(f"Default code chat template: {tokenizer.chat_template}")

    if custom_args['change_eos_token']:
        tokenizer.eos_token = custom_args['new_eos_token']

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|padding_token_[PAD]|>'})

    return tokenizer, instruction_template, response_template
