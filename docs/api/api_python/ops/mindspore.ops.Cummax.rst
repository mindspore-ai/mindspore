mindspore.ops.Cummax
=====================

.. py:class:: mindspore.ops.Cummax(axis)

    返回输入Tensor在指定轴上的累计最大值与其对应的索引。
    
    更多参考详见 :func:`mindspore.ops.cummax` 。

    参数：
        - **axis** (int) - 算子操作的维度，维度的大小范围是[-input.ndim, input.ndim - 1]。

    输入：
        - **input** (Tensor) - 输入Tensor。

    输出：
        一个包含两个Tensor的元组(values, indices)，分别表示累积最大值和对应索引。每个输出Tensor的shape和输入Tensor的shape相同。
