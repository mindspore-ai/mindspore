mindspore.mint.repeat_interleave
================================

.. py:function:: mindspore.mint.repeat_interleave(input, repeats, dim=None, *, output_size=None) -> Tensor

    沿着特定维度重复tensor的元素，类似 :func:`mindspore.numpy.repeat`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **repeats** (Union[int, tuple, list, Tensor]) - 指定复制次数，为正数。
        - **dim** (int, 可选) - 指定复制维度。默认 ``None`` ，此时输入tensor会被展平并且输出结果也会被展平。

    关键字参数：
        - **output_size** (int, 可选) - 给定维度的总输出大小（即参数 `repeats` 各元素之和）。默认 ``None`` 。

    返回：
        Tensor，值沿指定维度复制。如果输入的shape为 :math:`(s1, s2, ..., sn)` ，维度为i，则输出的shape为 :math:`(s1, s2, ..., si * repeats, ..., sn)` 。输出的数据类型与输入相同。
