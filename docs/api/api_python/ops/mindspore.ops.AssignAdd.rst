mindspore.ops.AssignAdd
=======================

.. py:class:: mindspore.ops.AssignAdd

    进行加法运算更新网络参数。

    更多细节请参考 :func:`mindspore.ops.assign_add` 。

    输入：
        - **variable** (Parameter) - 待更新的网络参数，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其轶应小于8。
        - **value** (Union[numbers.Number, Tensor]) - 待与 `variable` 相加的数值。如果类型为Tensor，则应与 `variable` 的shape相同。

    输出：
        Tensor，shape和dtype与 `variable` 相同。
