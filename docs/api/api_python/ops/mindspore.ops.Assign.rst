mindspore.ops.Assign
====================

.. py:class:: mindspore.ops.Assign

    为网络参数赋值。

    更多细节请参考 :func:`mindspore.ops.assign` 。

    输入：
        - **variable** (Parameter) - 待赋值的网络参数，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其轶应小于8。
        - **value** (Tensor) - 被赋给网络参数的值，和 `variable` 有相同的shape。

    输出：
        Tensor，shape和dtype与 `variable` 相同。
