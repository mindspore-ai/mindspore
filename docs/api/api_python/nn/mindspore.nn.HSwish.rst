mindspore.nn.HSwish
===================

.. py:class:: mindspore.nn.HSwish

    对输入的每个元素计算Hard Swish。input是具有任何有效形状的张量。

    Hard Swish定义如下：

    .. math::
        \text{hswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6},

    其中， :math:`x_{i}` 是输入的元素。

    输入：
        - **x** (Tensor) - 用于计算Hard Swish的Tensor。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度数。

    输出：
        Tensor，具有与 `x` 相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 的数据类型不支持。
