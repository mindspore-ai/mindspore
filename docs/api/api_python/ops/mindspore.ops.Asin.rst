mindspore.ops.Asin
==================

.. py:class:: mindspore.ops.Asin

    逐元素计算输入Tensor的反正弦值。

    更多细节请参考 :func:`mindspore.ops.asin` 。

    .. note::
        目前Ascend平台上不支持complex64和complex128类型输入。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(N,*)` 其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，shape与输入 `x` 相同。
