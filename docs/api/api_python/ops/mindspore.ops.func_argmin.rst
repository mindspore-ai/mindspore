mindspore.ops.argmin
====================

.. py:function:: mindspore.ops.argmin(input, axis=None, keepdims=False)

    返回tensor在指定轴上的最小值索引。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (Union[int, None]，可选) - 指定轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keepdims** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
