# 流管理介绍

## 设备流

设备流是属于特定设备的线性执行序列。默认情况下，每个设备都使用自己的“默认”流，不需要显式地创建一个设备流。

每个流中的操作都按照它们创建的顺序执行，但是不同流中的操作可以以任何相对顺序并发执行，除非使用显式同步函数（如`synchrize()`或`wait_stream()`）。例如，下面的代码是不正确的：

``` python
...
s = ms.hal.Stream()  # Create a new stream.
A = Tensor(np.random.randn(20, 20), ms.float32)
B = ops.matmul(A, A)
with ms.hal.StreamCtx(s):
    # sum() may start execution before matmul() finishes!
    C = ops.sum(B)
```

当 `current stream` 是默认流，切换至非默认流时，用户有责任确保正确的同步。正确的示例为：

``` python
...
s = ms.hal.Stream()  # Create a new stream.
A = Tensor(np.random.randn(20, 20), ms.float32)
B = ops.matmul(A, A)
s.wait_stream(ms.hal.current_stream())  # Dispatch wait event to device.
with ms.hal.StreamCtx(s):
    C = ops.sum(B)
```

由于申请和释放都需要开销，所以建议在同一个Cell或函数中使用同一个流，而不是每次使用前都申请一个流。不推荐的写法如下：

``` python
# The following cell formats are not recommended:
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        ...

    def construct(self, x):
        s1 = ms.hal.Stream()
        with ms.hal.StreamCtx(s1):
            ...

# The following function formats are not recommended:
def func(A):
    s1 = ms.hal.Stream()
    with ms.hal.StreamCtx(s1):
        ...
```

推荐的写法如下：

``` python
# The following cell formats are recommended:
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.s1 = ms.hal.Stream()
        ...

    def construct(self, x):
        with ms.hal.StreamCtx(self.s1):
            ...

# The following function formats are recommended:
s1 = ms.hal.Stream()
def func(A, s1):
    with ms.hal.StreamCtx(s1):
        ...
```

## 设备事件

设备事件可用于监视设备的进度、精确测量计时以及同步设备流。出于易用性考虑，我们在[Stream](https://www.mindspore.cn/docs/zh-CN/master/api_python/hal/mindspore.hal.Stream.html)类中提供了几个封装接口，比如上面例子中使用的 `wait_stream()` 接口，你可以通过接口文档了解详情。

`ms.hal.Event` 最常见的用法是 `record()` 和 `wait()` 的组合，以确保在多流网络中执行序能够满足用户的期望，如下所示：

``` python
s = ms.hal.Stream()     # Create a new stream.
ev1 = ms.hal.Event()    # Create a new event.
ev2 = ms.hal.Event()    # Create a new event.

A = Tensor(np.random.randn(20, 20), ms.float32)
B = ops.matmul(A, A)
ev1.record()
with ms.hal.StreamCtx(s):
    ev1.wait()  # Ensure 'B = ops.matmul(A, A)' is complete.
    C = ops.sum(B)
    ev2.record()
    ...
ev1.wait()      # Ensure 'C = ops.sum(B)' is complete.
D = C + 1
```

另外，由于Host算子的下发和设备算子的执行是异步的。在下面的例子中，Host上的 `add_s0` 持有的设备地址释放后，被 `mess_output` 申请到了，并将输出数据写入该设备地址。但是，此时设备可能会执行 `add_s1 = add_s0 + 1` ，但是 `add_s0` 的设备内存已经被其他算子使用了。最终导致精度不正确。

``` python
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, ops

context.set_context(mode=context.PYNATIVE_MODE)
s1 = ms.hal.Stream()

input_s0 = Tensor(np.random.randn(10, 32, 32, 32), ms.float32)
weight_s0 = Tensor(np.random.randn(32, 32, 3, 3), ms.float32)
mess_input = Tensor(np.random.randn(10, 32, 32, 32), ms.float32)

# Use default stream
add_s0 = input_s0 + 1
s1.wait_stream(ms.hal.default_stream())
with ms.hal.StreamCtx(s1):
  # Use conv2d because this operator takes a long time in device.
  conv_res = ops.conv2d(add_s0, weight_s0)
  add_s1 = add_s0 + 1
  ev = s1.record_event()
del add_s0

# Mess up the mem of add_s0. The value of add_s1 might be wrong.
mess_output = mess_input + 1
ev.wait()
add_1_s0 = add_s1 + 1
```

因此，MindSpore开发了一套自动延长跨流地址的生命周期的机制。你不需要手动执行类似 `record_stream()` 的函数来延长它的生命周期。总体原则是：MindSpore能够识别算子和输入输出所在的流，在识别出算子和输入输出跨流后（如 `add_s0` 来自stream0，算子在stream1上），对输入输出地址插入事件，并将地址和事件信息保存到显存池中。如果后续有“回流”（stream1上执行 `record()` ，stream0上执行 `wait()` )，则显存池中的地址和事件信息将被释放。如果没有“回流”，但是显存池分配失败，则所有事件将被同步并释放绑定的信息。
