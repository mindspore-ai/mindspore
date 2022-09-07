mindspore.ops.HSigmoid
=======================

.. py:class:: mindspore.ops.HSigmoid

    分段性逼近激活函数。

    逐元素计算。输入为任意维度的Tensor。

    HSigmoid定义为：

    .. math::
        \text{hsigmoid}(x_{i}) = max(0, min(1, \frac{x_{i} + 3}{6})),

    其中 :math:`x_i` 是输入Tensor的元素。

    输入：
        - **input_x** (Tensor) - 输入Tensor ，其shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，数据类型和shape与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
