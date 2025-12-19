mindspore.mint.argmin
=====================

.. py:function:: mindspore.mint.argmin(input, dim=None, keepdim=False)

    返回tensor在指定维度上的最小值索引。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, None]，可选) - 指定维度。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keepdim** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
