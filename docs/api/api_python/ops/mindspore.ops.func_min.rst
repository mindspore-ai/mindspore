mindspore.ops.min
==============================

.. py:function:: mindspore.ops.min(input, axis=0, keepdims=False, *, initial=None, where=None)

    在给定轴上计算输入Tensor的最小值。并且返回最小值和索引。

    .. note::
        在auto_parallel和semi_auto_parallel模式下，不能使用第一个输出索引。

    .. warning::
        - 如果有多个最小值，则取第一个最小值的索引。

    参数：
        - **input** (Tensor) - 输入任意维度的Tensor。不支持复数类型。
        - **axis** (int) - 指定计算维度。默认值：0。
        - **keepdims** (bool) - 表示是否减少维度，如果为True，输出将与输入保持相同的维度；如果为False，输出将减少维度。默认值：False。

    关键字参数：
        - **initial** (scalar, 可选) - 输出元素的最大值。如果 `input` 为空，则该参数必须设置。默认值：None。
        - **where** (Tensor[bool], 可选) - 表示是否需要将 `input` 中的原始值替换为 `initial` 值的Tensor。若为True则不替换，若为False则替换。`where` 中为False的位置，必须提供对应的 `initial` 值。默认值：None，表示默认为True。

    返回：
        tuple (Tensor)，表示2个Tensor组成的tuple，包含对应的索引和输入Tensor的最小值。

        - **output_x** (Tensor) - 输入Tensor的最小值，其shape与 `index` 相同，数据类型与 `input` 相同。
        - **index** (Tensor) - 输入Tensor最小值的索引，其数据类型为int32。如果 `keepdims` 为True，则输出Tensor的shape为 :math:`(input_1, input_2, ..., input_{axis-1}, 1, input_{axis+1}, ..., input_N)` 。否则，shape为 :math:`(input_1, input_2, ..., input_{axis-1}, input_{axis+1}, ..., input_N)` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `keepdims` 不是bool。
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `initial` 不是scalar。