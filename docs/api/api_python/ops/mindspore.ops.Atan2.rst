mindspore.ops.Atan2
===================

.. py:class:: mindspore.ops.Atan2

    逐元素计算x/y的反正切值。

    更多细节请参考 :func:`mindspore.ops.atan2` 。

    输入：
        - **x** (Tensor) - 输入Tensor，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **y** (Tensor) - 输入Tensor，shape应能在广播后与 `x` 相同，或 `x` 的shape在广播后与 `y` 相同。

    输出：
        Tensor，与广播后的输入shape相同，和 `x` 数据类型相同。
