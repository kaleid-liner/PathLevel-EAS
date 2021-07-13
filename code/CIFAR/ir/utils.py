import torch
from .constants import *


def get_tensor_shape(tensor):
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)
    if len(shape) == 4:
        shape = [shape[0], shape[2], shape[3], shape[1]]
    return shape


def apply_mask(G, masks_file):
    masks = torch.load(masks_file)
    print(G.nodes)
    for name in G.nodes:
        if G.nodes[name] and G.nodes[name]['op'] == 'conv':
            G.nodes[name]['cin'] = int(G.nodes[name]['cin'] * 0.5)
            G.nodes[name]['cout'] = int(G.nodes[name]['cout'] * 0.5)
    # for name, mask in masks.items():
    #     if name in G.nodes and G.nodes[name]['op'] == CONV_TYPE:
    #         channel_mask = (mask['weight'].abs().sum((1, 2, 3)) != 0).int()
