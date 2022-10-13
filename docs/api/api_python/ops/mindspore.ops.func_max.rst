mindspore.ops.max
=================

.. py:function:: mindspore.ops.max(x, axis=0, keep_dims=False)

    根据指定的索引计算最大值，并返回索引和值。

    在给定轴上计算输入Tensor的最大值。并且返回最大值和索引。

    .. note::
        在auto_parallel和semi_auto_parallel模式下，不能使用第一个输出索引。

    .. warning::
        - 如果有多个最大值，则取第一个最大值的索引。
        - "axis"的取值范围为[-dims, dims - 1]。"dims"为"x"的维度长度。

    参考：:class: `mindspore.ops.ArgMaxWithValue`。

    参数：
        - **x** (Tensor) - 输入任意维度的Tensor。将输入Tensor的shape设为 :math:`(x_1, x_2, ..., x_N)` 。数据类型为mindspore.float16或float32。
        - **axis** (int) - 指定计算维度。默认值：0。
        - **keep_dims** (bool) - 表示是否减少维度，如果为True，输出将与输入保持相同的维度；如果为False，输出将减少维度。默认值：False。

    返回：
        tuple (Tensor)，表示2个Tensor组成的tuple，包含对应的索引和输入Tensor的最大值。

        - **index** (Tensor) - 输入Tensor最大值的索引。如果 `keep_dims` 为True，则输出Tensor的shape为 :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)` 。否则，shape为 :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` 。
        - **output_x** (Tensor) - 输入Tensor的最大值，其shape与索引相同。

    异常：
        - **TypeError** - `input_x` 的数据类型非float16、float32或float64。
        - **TypeError** - `keep_dims` 不是bool。
        - **TypeError** - `axis` 不是int。