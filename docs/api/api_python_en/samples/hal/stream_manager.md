# Illustration of stream management

## Device stream

A device stream is a linear sequence of execution that belongs to a specific device. You normally do not need to create one explicitly: by default, each device uses its own "default" stream.

Operations in each stream are serialized in the order in which they are created, but operations from different streams can execute concurrently in any relative order, unless explicit synchronization functions (such as `synchronize()` or `wait_stream()`) are used. For example, the following code is incorrect:

```python
...
s = ms.hal.Stream()  # Create a new stream.
A = Tensor(np.random.randn(20, 20), ms.float32)
B = ops.matmul(A, A)
with ms.hal.StreamCtx(s):
    # sum() may start execution before matmul() finishes!
    C = ops.sum(B)

```

When the `current stream` is the default stream, it is the user’s responsibility to ensure proper synchronization when switch to the non-default streams. The fixed version of this example is:

```python
...
s = ms.hal.Stream()  # Create a new stream.
A = Tensor(np.random.randn(20, 20), ms.float32)
B = ops.matmul(A, A)
s.wait_stream(ms.hal.current_stream())  # Dispatch wait event to device.
with ms.hal.StreamCtx(s):
    C = ops.sum(B)
```

Because both application and release require overheads, it is recommended that the same stream be used in the same cell or function instead of applying for a stream before each use. The following code is not recommended:

```python
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

The recommended format is as follows:

```python
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

## Device event

Device events can be used to monitor the device’s progress, to accurately measure timing, and to synchronize device streams. For ease of use, we provide several encapsulation interfaces in the [Stream](https://www.mindspore.cn/docs/en/master/api_python/hal/mindspore.hal.Stream.html) class, such as the `wait_stream()` interface used in the above example, you can see the interface documentation for details.

The most common use of `ms.hal.Event` is the combination of `record()` and `wait()` to ensure that the execution order can meet the user's expectations in a multi-stream network, as follows:

```python
s = ms.hal.Stream()     # Create a new stream.
ev1 = ms.hal.Event()    # Create a new event.
ev2 = ms.hal.Event()    # Create a new event.

A = Tensor(np.random.randn(20, 20), ms.float32)
B = ops.matmul(A, A)
ev1.record()
with ms.hal.StreamCtx(s):
    ev1.wait()  # Ensure 'B = ops.matmul(A, A)' is complete.
    C = ops.sum(B)
    ev2.record()
    ...
ev1.wait()      # Ensure 'C = ops.sum(B)' is complete.
D = C + 1
```

In addition, delivery of the host operator and execution of the device operator are asynchronous. In the following example, after the device address of `add_s0` on the host is released, it is applied for by `mess_output`, and the output data is written to the device address. However, at this time, the device may execute `add_s1 = add_s0 + 1`, but the device memory of `add_s0` has been used by other operators. As a result, the precision is incorrect.

```python
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

Therefore, MindSpore has developed a mechanism to automatically extend the lifecycle of cross-streams device addresses. You don't need to manually execute the function like `record_stream()` to extend its lifetime. The general principle is as follows: MindSpore identifies the stream where the operator and input/output are located. After identifying the operator and input/output cross-streams(For example, `add_s0` is from stream0, and the operator is on stream1), an event will be inserted for the input or output address, and the address and event information is saved to the device memory pool. Then, if there is a "back-stream"(`record()` on stream1, `wait()` on stream0), the address and event information in the device memory pool will be removed. If there is no "back-stream" and the device memory pool fails to allocate, all events will be synchronized and the bound information is released.
