mindspore.ops.IsNan
====================

.. py:class:: mindspore.ops.IsNan

    判断输入数据每个位置上的值是否是NaN。

    更多参考详见 :func:`mindspore.ops.isnan`。

    输入：
        - **x** (Tensor) - IsNan的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，shape与相同的输入，数据的类型为bool。
