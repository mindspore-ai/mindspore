import mindspore as ms
from mindspore import mutable
from mindspore import context


@ms.jit
def func(input_x, input_y, t):
    output = input_x
    for _ in range(2):
        output = input_x + input_x * input_y + output
    return output, t


context.set_context(precompile_only=True)
x = ms.Tensor([1], ms.dtype.float32)
y = ms.Tensor([2], ms.dtype.float32)
t1 = mutable((1,), dynamic_len=True)
t2 = mutable((1, 2,), dynamic_len=True)
out1 = func(x, y, t1)
out2 = func(x, y, t2)
print("out1:", out1)
print("out2:", out2)
context.set_context(precompile_only=False)
