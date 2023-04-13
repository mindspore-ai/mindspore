mindspore.ops.BitwiseAnd
========================

.. py:class:: mindspore.ops.BitwiseAnd

    逐元素执行两个Tensor的与运算。

    更多细节详见 :func:`mindspore.ops.bitwise_and`。

    输入：
        - **x** (Tensor) - 第一个输入Tensor，其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **y** (Tensor) - 第二个输入Tensor，数据类型与 `x` 一致。

    输出：
        Tensor，是一个与 `x` 相同类型的Tensor。
