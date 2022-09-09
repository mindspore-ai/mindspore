mindspore.Tensor.broadcast_to
=============================

.. py:method:: mindspore.Tensor.broadcast_to(shape)

    将输入shape广播到目标shape。

    更多细节请参考 :func:`mindspore.ops.broadcast_to` 。

    参数：
        - **shape** (tuple) - 要广播的目标形状。可以由用户指定，或在要广播的维度上指定-1，它将被该位置的输入张量形状替换。

    返回：
        Tensor，形状为用户指定的 `shape`，类型和 `self` 相同。

    异常：
        - **TypeError** - 如果输入的 `shape` 参数不是tuple类型。
        - **ValueError** - 如果输入的 `shape` 与 `self` 的形状不兼容，或者目标 `shape` 中的-1位于无效位置。