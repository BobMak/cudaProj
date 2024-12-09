import onnx
import torch as th
import torch.onnx
from torchvision.models import resnet18

import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relay.frontend import from_onnx


model = resnet18()

# Save the model
th.save(model.state_dict(), 'model.pth')
# export to onnx
torch.onnx.export(model, th.randn(1, 3, 224, 224), 'model.onnx', export_params=True)
onnx_model = onnx.load("model.onnx")
# convert to tvm

model, params = from_onnx(onnx_model)
model.show()

