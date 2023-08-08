mindspore.ops.Atan2
===================

.. py:class:: mindspore.ops.Atan2

    逐元素计算x/y的反正切值。

    更多细节请参考 :func:`mindspore.ops.atan2` 。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(N,*)` 其中 :math:`*` 表示任意数量的附加维度。
        - **y** (Tensor) - 第二个输入Tensor，shape与 `x` 相同，或能与 `x` 的shape广播。

    输出：
        Tensor，与广播后的输入shape相同，和 `x` 数据类型相同。
