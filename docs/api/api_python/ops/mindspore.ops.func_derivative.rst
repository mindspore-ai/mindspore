mindspore.ops.derivative
========================

.. py:function:: mindspore.ops.derivative(fn, primals, order)

    计算函数或网络输出对输入的高阶微分。给定待求导函数的原始输入和求导的阶数n，将返回函数输出对输入的第n阶导数。输入的初始1阶导数在内部默认设置为1，其他阶设置为0。

    .. note::
        - 若 `primals` 是int型的Tensor，会被转化成float32格式进行计算。

    参数：
        - **fn** (Union[Cell, function]) - 待求导的函数或网络。
        - **primals** (Union[Tensor, tuple[Tensor]]) - `fn` 的输入，单输入的type为Tensor，多输入的type为Tensor组成的tuple。
        - **order** (int) - 求导的阶数。

    返回：
        tuple，由 `out_primals` 和 `out_series` 组成。

        - **out_primals** (Union[Tensor, list[Tensor]]) - `fn(primals)` 的结果。
        - **out_series** (Union[Tensor, list[Tensor]]) - `fn` 输出对输入的第n阶导数。

    异常：
        - **TypeError** - `primals` 不是Tensor或tuple。
        - **TypeError** - `order` 不是int。
        - **ValueError** - `order` 不是正数。
