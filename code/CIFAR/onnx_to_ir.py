import onnx
from onnx import shape_inference
from ir.converters import OnnxConverter
import json
import networkx as nx
from ir.utils import apply_mask


model = onnx.load('/home/ticao/v-wjiany/pytorch-onnx-experiments/pytorch/onnx/mobilenet_v2_opt.onnx')
inferred_model = shape_inference.infer_shapes(model)
graph = inferred_model.graph

for node in graph.node:
    print(node)

parser = OnnxConverter(graph)

G = parser.G

with open('mobilenetv2_grapher.json', 'w') as f:
    json.dump(parser.to_grapher(), f, indent=4)

with open('mobilenetv2_kernels.json', 'w') as f:
    json.dump(parser.to_kernels(), f, indent=4)

'''
apply_mask(G, '/home/ticao/v-wjiany/PathLevel-EAS/Nets/CIFAR10#PyramidTreeCellA#N=18_alpha=84#300/checkpoint/pruned_mask_agp.pth')

with open('pyramidnet_pruned.json', 'w') as f:
    json.dump(dict(G.nodes.data()), f)

L = nx.line_graph(G)

with open('pyramidnet_pruned_oplatency.json', 'r') as f:
    latency = json.load(f)

for edge in L.edges:
    assert(edge[0][1] == edge[1][0])
    node = edge[0][1]
    L.edges[edge]['weight'] = latency.get(node, 0)

print(nx.dag_longest_path_length(L), sum(latency.values()))
'''