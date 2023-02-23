mindspore.ops.count_nonzero
============================

.. py:function:: mindspore.ops.count_nonzero(x, dims=None)

    计算输入Tensor指定轴上的非零元素的数量。如果没有指定维度，则计算Tensor中所有非零元素的数量。

    .. note::
        `dims` 的值范围是[-x_dims，x_dims)。其中， `x_dims` 是输入 `x` 的维度。

    参数：
        - **x** (Tensor) - 要计算的输入，可以是任意维度的Tensor。将输入张量的shape设为 :math:`(x_1, x_2, ..., x_N)` 。
        - **dims** (Union[int, list(int), tuple(int)]，可选) - 要沿其计算非零值数量的维度。默认值：None。

    返回：
        一个N维Tensor，表示输入Tensor在 `dims` 上的非零元素数量。 `dims` 指定的维度将被规约掉。例如，如果 `x` 的大小为 :math:`(2, 3, 4)` ， `dims` 为 :math:`[0, 1]` ，则y_shape将为 :math:`(4,)` 。

    异常：
        - **TypeError** - 如果 `x` 的数据类型不受支持。
        - **TypeError** - 如果 `dims` 的数据类型不是int。
        - **ValueError** - 如果 `dims` 中的任何值不在 :math:`[-x_dims，x_dims)` 范围内。
