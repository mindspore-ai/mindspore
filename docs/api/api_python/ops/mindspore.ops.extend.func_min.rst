mindspore.ops.extend.min
========================

.. py:function:: mindspore.ops.extend.min(input, dim=None, keepdim=False)

    在给定轴上计算输入Tensor的最小值。并且返回最小值和索引。

    参数：
        - **input** (Tensor) - 输入任意维度的Tensor。不支持复数类型。
        - **dim** (int, 可选) - 指定计算维度。默认值： ``None`` 。
        - **keepdim** (bool, 可选) - 表示是否减少维度，如果为 ``True`` ，输出将与输入保持相同的维度；如果为 ``False`` ，输出将减少维度。默认值： ``False`` 。

    返回：
        如果 `dim` 不为 ``None`` 返回值为Tensor，其值为输入Tensor的最小值，其shape为 :math:`()` ， 其数据类型与 `input` 相同。

        如果 `dim` 为 ``None`` 返回值为为tuple (Tensor)，表示2个Tensor组成的tuple，包含输入Tensor在给定轴上的最小值和对应的索引：

        - **values** (Tensor) - 输入Tensor在给定轴上的最小值，其数据类型与 `input` 相同。如果 `keepdim` 为 ``True`` ，则输出Tensor的shape为 :math:`(input_1, input_2, ..., input_{dim-1}, 1, input_{dim+1}, ..., input_N)` 。否则，shape为 :math:`(input_1, input_2, ..., input_{dim-1}, input_{dim+1}, ..., input_N)` 。
        - **index** (Tensor) - 输入Tensor在给定轴上的最小值的索引，其shape与 `values` 相同。

    异常：
        - **ValueError** - 如果 `dim` 为 ``None`` ，但 `keepdim` 不为 ``False`` 。
