mindspore.ops.extend.max
========================

.. py:function:: mindspore.ops.extend.max(input, dim, keepdim=False)

    在给定轴上计算输入Tensor的最大值。并且返回最大值和索引。

    参数：
        - **input** (Tensor) - 输入任意维度的Tensor。不支持复数类型。
        - **axis** (int) - 指定计算维度。
        - **keepdim** (bool) - 表示是否减少维度，如果为 ``True`` ，输出将与输入保持相同的维度；如果为 ``False`` ，输出将减少维度。默认值： ``False`` 。

    返回：
        tuple (Tensor)，表示2个Tensor组成的tuple，包含输入Tensor的最大值和对应的索引。

        - **values** (Tensor) - 输入Tensor的最大值，其数据类型与 `input` 相同。如果 `keepdim` 为True，则输出Tensor的shape为 :math:`(input_1, input_2, ..., input_{axis-1}, 1, input_{axis+1}, ..., input_N)` 。否则，shape为 :math:`(input_1, input_2, ..., input_{axis-1}, input_{axis+1}, ..., input_N)` 。
        - **index** (Tensor) - 输入Tensor最大值的索引， 其shape与 `values` 相同。
