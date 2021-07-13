import onnx
import networkx as nx
from ir.utils import get_tensor_shape
from .constants import *
from itertools import chain


class OnnxConverter:
    def __init__(self, onnx_graph):
        self.graph = onnx_graph

        self.tensors = {}
        for tensor in chain(self.graph.input, self.graph.value_info, self.graph.output):
            self.tensors[tensor.name] = {
                'shape': get_tensor_shape(tensor),
                'inputs': [],
                'outputs': [],
            }

        for node in self.graph.node:
            for input_name in node.input:
                if input_name in self.tensors:
                    self.tensors[input_name]['outputs'].append(node)
            for output_name in node.output:
                if output_name in self.tensors:
                    self.tensors[output_name]['inputs'].append(node)

        self.G = self.to_networkx()

    def to_networkx(self):
        G = nx.DiGraph()

        sliced_tensors = set()
        selected_slice = set()
        for node in self.graph.node:
            if node.op_type == SLICE_TYPE:
                tensor = node.input[0]
                if tensor in sliced_tensors:
                    continue
                else:
                    sliced_tensors.add(tensor)
                    selected_slice.add(node.name)
            G.add_node(node.name, **self.fetch_attrs(node))

        for node in self.graph.node:
            if node.op_type == SLICE_TYPE and node.name not in selected_slice:
                continue
            for input_name in node.input:
                if input_name in self.tensors: # remove dummy ops
                    G.add_edge(input_name, node.name)
            for output_name in node.output:
                if output_name in self.tensors:
                    G.add_edge(node.name, output_name)
                if node.op_type == SLICE_TYPE:
                    for tensor_name in self._get_sibling_slice_output_tensors(node):
                        G.add_edge(node.name, tensor_name)

        return G

    def fetch_attrs(self, node):
        attrs = {}
        input_tensors = []
        for input_name in node.input:
            if input_name in self.tensors:
                input_tensors.append(self.tensors[input_name]['shape'])
        output_tensors = []
        for output_name in node.output:
            if output_name in self.tensors:
                output_tensors.append(self.tensors[output_name]['shape'])
        if node.op_type == SLICE_TYPE:
            for tensor_name in self._get_sibling_slice_output_tensors(node):
                output_tensors.append(self.tensors[tensor_name]['shape'])
        if len(input_tensors) == 0 or len(input_tensors[0]) <= 1 or len(output_tensors) == 0 or len(output_tensors[0]) <= 1:
            return attrs

        if node.op_type not in OP_ALIAS:
            print(f'Unsupported OP: {node.op_type}')
            return attrs

        attrs['op'] = OP_ALIAS[node.op_type]
        attrs['input_tensors'] = input_tensors
        attrs['output_tensors'] = output_tensors
        if node.op_type in [CONV_TYPE, MAXPOOL_TYPE, AVGPOOL_TYPE]:
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    attrs['ks'] = list(attr.ints)
                elif attr.name == 'strides':
                    attrs['strides'] = list(attr.ints)
                elif attr.name == 'pads':
                    attrs['padding'] = list(attr.ints)
        elif node.op_type == SLICE_TYPE:
            for attr in node.attribute:
                if attr.name == 'axes':
                    attrs['split_dim'] = list(attr.ints)
        elif node.op_type == FC_TYPE:
            attrs['cin'] = input_tensors[0][1]
            attrs['cout'] = output_tensors[0][1]

        if len(input_tensors) == 1 and len(input_tensors[0]) == 4:
            attrs['inputh'] = input_tensors[0][1]
            attrs['inputw'] = input_tensors[0][2]
            attrs['cin'] = input_tensors[0][3]

        if len(output_tensors) == 1 and len(output_tensors[0]) == 4:
            attrs['cout'] = output_tensors[0][3]

        return attrs

    def to_kernels(self):
        return [n for n in self.G.nodes.values() if n]

    def to_grapher(self):
        result = {}

        for node in self.G.nodes:
            node_attrs = self.G.nodes[node]
            if node in self.tensors or not node_attrs:
                continue

            attr = {'attr': {}}
            attr['input_shape'] = node_attrs['input_tensors']
            attr['output_shape'] = node_attrs['output_tensors']
            attr['type'] = node_attrs['op']

            outbounds = []
            inbounds = []
            for successor in self.G.successors(node):
                try:
                    outbounds.append(next(self.G.successors(successor)))
                except StopIteration:
                    pass
            for predecessor in self.G.predecessors(node):
                try:
                    inbounds.append(next(self.G.predecessors(predecessor)))
                except StopIteration:
                    pass

            if node_attrs['op'] in [OP_ALIAS[CONV_TYPE], OP_ALIAS[AVGPOOL_TYPE], OP_ALIAS[MAXPOOL_TYPE]]:
                attr['attr']['strides'] = node_attrs['strides']
                attr['attr']['padding'] = node_attrs['padding']
                attr['attr']['weight_shape'] = node_attrs['ks'] + [node_attrs['cin'], node_attrs['cout']]
            elif node_attrs['op'] == OP_ALIAS[SLICE_TYPE]:
                attr['attr']['split_dim'] = node_attrs['split_dim']

            result[node] = {
                'attr': attr,
                'outbounds': outbounds,
                'inbounds': inbounds,
            }

        return result

    def _get_sibling_slice_output_tensors(self, node):
        output_tensors = []
        for slice in self.tensors[node.input[0]]['outputs']:
            if slice.name != node.name and slice.op_type == SLICE_TYPE:
                for output_name in slice.output:
                    if output_name in self.tensors:
                        output_tensors.append(output_name)

        return output_tensors
