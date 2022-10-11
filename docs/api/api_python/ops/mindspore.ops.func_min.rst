mindspore.ops.min
==============================

.. py:function:: mindspore.ops.min(x, axis=0, keep_dims=False)

    根据指定的索引计算最小值，并返回索引和值。

    在给定轴上计算输入Tensor的最小值。并且返回最小值和索引。

    .. note::
        在auto_parallel和semi_auto_parallel模式下，不能使用第一个输出索引。

    .. warning::
        - 如果有多个最小值，则取第一个最小值的索引。
        - "axis"的取值范围为[-dims, dims - 1]。"dims"为"x"的维度长度。

    参数：
        - **x** (Tensor) - 输入任意维度的Tensor。将输入Tensor的shape设为 :math:`(x_1, x_2, ..., x_N)` 。数据类型为mindspore.uint16，mindspore.uint32，mindspore.int16，mindspore.int32，mindspore.float16或者mindspore.float32。
        - **axis** (int) - 指定计算维度。默认值：0。
        - **keep_dims** (bool) - 表示是否减少维度，如果为True，输出将与输入保持相同的维度；如果为False，输出将减少维度。默认值：False。

    返回：
        tuple (Tensor)，表示2个Tensor组成的tuple，包含对应的索引和输入Tensor的最小值。

        - **index** (Tensor) - 输入Tensor最小值的索引。如果 `keep_dims` 为True，则输出Tensor的shape为 :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)` 。否则，shape为 :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` 。
        - **output_x** (Tensor) - 输入Tensor的最小值，其shape与索引相同。

    异常：
        - **TypeError** - `input_x` 的数据类型非uint16，uint32，int16，int32，float16，float32。
        - **TypeError** - `keep_dims` 不是bool。
        - **TypeError** - `axis` 不是int。