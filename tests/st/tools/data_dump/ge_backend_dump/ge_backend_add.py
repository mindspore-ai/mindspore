import os
import numpy as np
import mindspore as ms

os.environ['HOOK_TOOL_PATH'] = './build/lib/libmsprobe_debug_stub.so'
os.environ['MS_HOOK_ENABLE'] = 'on'
os.environ['ENABLE_MS_GE_DUMP'] = '1'
os.environ['MINDSPORE_DUMP_CONFIG'] = "fake_path"
x_np = np.random.randn(3, 3, 3)
y_np = np.random.randn(3, 3, 3)
x_tensor = ms.Tensor(x_np, ms.bfloat16)
y_tensor = ms.Tensor(y_np, ms.bfloat16)

class Net(ms.nn.Cell):
    @ms.jit(backend="GE")
    def construct(self, x, y):
        z = ms.ops.add(x, y)
        return z

net = Net()

out = net(x_tensor, y_tensor)

print("exec success! output is: \n", out)
