import json

import torch
import torch.nn as nn


def load_freeze(model, pth_path):
    # load from pretrained QGeoGNN params, freeze the GNN encoder part.
    tlc_params = torch.load(pth_path)
    tlc_params_encoder = {
        k: v for k, v in tlc_params.items() if "gnn_node" in k
    }
    # print(model.state_dict().keys())
    model.state_dict().update(tlc_params_encoder)

    # freeze the encoder layers
    for layer_name, param in model.named_parameters():
        if "gnn_node" in layer_name:
            param.requires_grad = False