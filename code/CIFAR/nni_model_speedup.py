import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.algorithms.compression.pytorch.pruning import LevelPruner, L1FilterPruner
from nni.compression.pytorch import apply_compression_results, ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.retiarii.converter.graph_gen import convert_to_graph

import time
from torchviz import make_dot
from torchsummary import summary
from torchvision.models import densenet201, densenet161


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 180, 7)
        self.conv2 = nn.Conv2d(180, 90, 7)
        self.conv3 = nn.Conv2d(180, 90, 7)
        self.conv4 = nn.Conv2d(180, 180, 7)
        # an affine operation: y = Wx + b

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x1 = self.conv4(x)
        x2 = torch.cat((self.conv2(x), self.conv3(x)), 1)
        return (x1 + x2)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = densenet161()
cuda_model = densenet161().cuda()
inputs = torch.randn(1, 3, 224, 224).cuda()
for i in range(5):
    cuda_model(inputs)
cuda_model = densenet161().cuda()
inputs = torch.randn(1, 3, 224, 224).cuda()
now = time.time()
# script = torch.jit.script(model)
# cuda_model(inputs)
trace = torch.jit.trace(model, example_inputs=torch.randn(1, 3, 224, 224))

print(time.time() - now)

# for node in script.graph.nodes():
#     if node.hasAttribute('name') and node.s('name').endswith('forward'):
#         submodule = node.inputsAt(0).node()
#         print(submodule.s('name'))


'''
dummy_input = torch.randn(1, 3, 32, 32).cuda()

model = torch.load('pyramidnet.pt').cuda()
# torch.onnx.export(model, dummy_input, 'pyramidnet.onnx', verbose=True)
mask_path = '../../Nets/CIFAR10#PyramidTreeCellA#N=18_alpha=84#300/checkpoint/pruned_mask_agp.pth'
# make_dot(model(dummy_input), params=dict(model.named_parameters())).view()

model.eval()
for _ in range(32):
    use_mask_out = model(dummy_input)
start = time.time()
for _ in range(32):
    use_mask_out = model(dummy_input)
print('elapsed time when use mask: ', time.time() - start)
apply_compression_results(model, mask_path, 'cuda')

flops, params, results = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params}")

for _ in range(32):
    use_mask_out = model(dummy_input)
start = time.time()
for _ in range(32):
    use_mask_out = model(dummy_input)
print('elapsed time when use mask: ', time.time() - start)

m_speedup = ModelSpeedup(model, dummy_input, mask_path, 'cuda')
m_speedup.speedup_model()

flops, params, results = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params}")

for _ in range(32):
    use_mask_out = model(dummy_input)
start = time.time()
for _ in range(32):
    use_speedup_out = model(dummy_input)
print('elapsed time when use speedup: ', time.time() - start)
'''