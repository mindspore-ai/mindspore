mindspore.mint.median
======================

.. py:function:: mindspore.mint.median(input, dim=None, keepdim=False)

    输出指定维度上的中值与其对应的索引。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int, 可选) - 指定计算维度。默认 ``None`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        两个tensor组成的tuple(median, median_indices)
