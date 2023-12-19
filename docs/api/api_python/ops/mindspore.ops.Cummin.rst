mindspore.ops.Cummin
=====================

.. py:class:: mindspore.ops.Cummin(axis)

    返回输入Tensor在指定轴上的累积最小值与其对应的索引。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.cummin` 。

    参数：
        - **axis** (int) - 算子操作的维度，维度的大小范围是[-input.ndim, input.ndim - 1]。

    输入：
        - **input** (Tensor) - 输入Tensor。

    输出：
        一个包含两个Tensor的元组(values, indices)，分别表示累积最小值和对应索引。每个输出Tensor的shape和输入Tensor的shape相同。
