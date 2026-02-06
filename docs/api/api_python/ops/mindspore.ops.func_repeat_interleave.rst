mindspore.ops.repeat_interleave
================================

.. py:function:: mindspore.ops.repeat_interleave(input, repeats, axis=None)

    在指定轴上复制输入tensor的元素，类似 :func:`mindspore.numpy.repeat`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **repeats** (Union[int, tuple, list, Tensor]) - 复制次数，为正数。
        - **axis** (int，可选) - 指定轴，默认 ``None`` 。如果为 ``None`` ，输入和输出tensor会被展平为一维。

    返回：
        Tensor
