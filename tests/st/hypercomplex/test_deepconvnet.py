import numpy as np
from mindspore import context, Tensor
from mindspore.ops import operations as P
from deepconvnet import DeepConvNet


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    model = DeepConvNet()
    model.set_train(False)
    u = Tensor(np.random.random((2, 32, 1, 4096)).astype(np.float32))
    y = model(u)
    print(P.Shape()(y), y)
