from copy import deepcopy
import re


import torch


def freeze_old_model_layers(model, list_of_layers=None):
    if list_of_layers is None:
        list_of_layers = set()
        for k in model.state_dict().keys():
            i = re.search('layers.[0-9]+.', k)
            if i is not None:
                i = int(i[0][len('layers.'):-1])
            else:
                i = None
            if 'down_proj' in k or 'o_proj' in k:
                if torch.equal(torch.zeros_like(model.state_dict()[k]), model.state_dict()[k]):
                    list_of_layers.add(i)
    for param in model.base_model.embed_tokens.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.base_model.layers):
        if i not in list_of_layers:
            for param in layer.parameters():
                param.requires_grad = False
    for param in model.base_model.norm.parameters():
        param.requires_grad = False
    return model


def add_new_layers(model, state_dict_path, freeze_old_layers=True):
    st_dict = torch.load(state_dict_path)

    list_of_layers = set()
    for k in st_dict.keys():
        i = re.search('layers.[0-9]+.', k)
        if i is not None:
            i = int(i[0][len('layers.'):-1])
        else:
            i = None
        if 'down_proj' in k or 'o_proj' in k:
            if torch.equal(torch.zeros_like(st_dict[k]), st_dict[k]):
                list_of_layers.add(i)
    for new_layer in sorted(list(list_of_layers)):
        model.base_model.layers.insert(new_layer, deepcopy(model.model.layers[new_layer-1]))

    model.load_state_dict(st_dict)
    model.config.num_hidden_layers = model.config.num_hidden_layers + len(list_of_layers)

    if freeze_old_layers:
        model = freeze_old_model_layers(model, list_of_layers)

    return model
