mindspore.Tensor.all
====================

.. py:method:: mindspore.Tensor.all(axis=None, keep_dims=False) -> Tensor

    检查指定维度上是否所有元素均为 ``True``。

    参数：
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 要减少的维度。如果为 ``None`` ，减少所有维度。默认 ``None`` 。
        - **keep_dims** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor

    .. py:method:: mindspore.Tensor.all(dim=None, keepdim=False) -> Tensor
        :noindex:

    检查指定维度上是否所有元素均为 ``True``。

    参数：
        - **dim** (Union[int, tuple(int), list(int), Tensor], 可选) - 要减少的维度。如果为 ``None`` ，减少所有维度。默认 ``None`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
