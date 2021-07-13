from nni.retiarii.graph import Graph
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii.converter.graph_gen import convert_to_graph, GraphConverter

import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn

from torchsummary import summary

class MyModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(32, 1, 5)
    self.pool = nn.MaxPool2d(kernel_size=2)
  def forward(self, x):
    return self.pool(self.conv(x))

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.mymodule = MyModule()
    nn.Sequential()
  def forward(self, x):
    return F.relu(self.mymodule(x))

class VGG(nn.Module):
  def __init__(
      self,
      features: nn.Module,
      num_classes: int = 1000,
      init_weights: bool = True
  ) -> None:
      super(VGG, self).__init__()
      self.features = features
      self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
      self.classifier = nn.Sequential(
          nn.Linear(512 * 7 * 7, 4096),
          nn.ReLU(True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(True),
          nn.Dropout(),
          nn.Linear(4096, num_classes),
      )
      if init_weights:
          self._initialize_weights()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      x = self.features(x)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x

model = Model()
script = torch.jit.script(model)
names = {}

converter = GraphConverter()
convert_to_graph(script, model).root_graph._dump()
summary(model, input_size=(32, 224, 224), device='cpu')

for node in script.graph.nodes():
    if node.hasAttribute('name') and node.s('name').endswith('forward'):
        submodule = node.inputsAt(0).node()
        print(submodule.s('name'))
