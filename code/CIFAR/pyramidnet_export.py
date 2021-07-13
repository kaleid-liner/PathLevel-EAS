from models.run_manager import get_model_by_name
import json
import torch
import torch.nn as nn


net_config = '/home/ticao/v-wjiany/PathLevel-EAS/Nets/CIFAR10#PyramidTreeCellA#N=18_alpha=84#300/net.config'
with open(net_config, 'r') as f:
    model = get_model_by_name('PyramidNet').set_from_config(json.load(f))

dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, 'pyramidnet.onnx', verbose=True)
