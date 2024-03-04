mindspore.ops.ArgMaxWithValue
==============================

.. py:class:: mindspore.ops.ArgMaxWithValue(axis=0, keep_dims=False)

    在给定轴上计算输入Tensor的最大值，并且返回最大值和索引。

    .. note::
        在auto_parallel和semi_auto_parallel模式下，不能使用第一个输出索引。

    .. warning::
        - 如果有多个最大值，则取第一个最大值的索引。
        - `axis` 的取值范围为[-dims, dims - 1]。"dims"为 `input` 的维度长度。

    参考： :func:`mindspore.ops.max` 。

    参数：
        - **axis** (int) - 指定计算维度。默认值： ``0`` 。
        - **keep_dims** (bool) - 表示是否减少维度，如果为 ``True`` ，输出将与输入保持相同的维度；如果为 ``False`` ，输出将减少维度。默认值： ``False`` 。

    输入：
        - **input** (Tensor) - 输入任意维度的Tensor。将输入Tensor的shape设为 :math:`(input_1, input_2, ..., input_N)` 。

    输出：
        tuple (Tensor)，表示2个Tensor组成的tuple，包含对应的索引和输入Tensor的最大值。

        - **index** (Tensor) - 输入Tensor最大值的索引，数据类型为int64。如果 `keep_dims` 为 ``True`` ，则输出Tensor的shape为 :math:`(input_1, input_2, ..., input_{axis-1}, 1, input_{axis+1}, ..., input_N)` 。否则，shape为 :math:`(input_1, input_2, ..., input_{axis-1}, input_{axis+1}, ..., input_N)` 。
        - **values** (Tensor) - 输入Tensor的最大值，其shape与 `index` 相同，数据类型与 `input` 相同。

    异常：
        - **TypeError** - `keep_dims` 不是bool。
        - **TypeError** - `axis` 不是int。
