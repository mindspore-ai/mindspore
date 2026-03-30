mindspore.Tensor.argmax
=======================

.. py:method:: mindspore.Tensor.argmax(axis=None, keepdims=False) -> Tensor

    返回tensor在指定轴上的最大值索引。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：删除 `argmax(axis=None, keepdims=False)` 重载，仅支持 `argmax(dim=None, keepdim=False)`。

    参数：
        - **axis** (Union[int, None]，可选) - 指定计算轴。如果为 ``None`` ，计算tensor中的所有元素。默认 ``None`` 。
        - **keepdims** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor

    .. py:method:: mindspore.Tensor.argmax(dim=None, keepdim=False) -> Tensor
        :noindex:

    返回tensor在指定轴上的最大值索引。

    参数：
        - **dim** (Union[int, None]，可选) - 指定计算维度。如果为 ``None`` ，计算tensor中的所有元素。默认 ``None`` 。
        - **keepdim** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
