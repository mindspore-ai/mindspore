mindspore.ops.argmax
====================

.. py:function:: mindspore.ops.argmax(input, dim=None, keepdim=False)

    返回tensor在指定维度上的最大值索引。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, None]，可选) - 指定维度。如果为 ``None`` ，则计算 `input` 中所有元素的最大值索引。默认 ``None`` 。
        - **keepdim** (bool，可选) - 是否保留输出tensor的维度。默认 ``False`` 。

    返回：
        Tensor
