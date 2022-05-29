mindspore.ops.Softsign
======================

.. py:class:: mindspore.ops.Softsign

    Softsign激活函数。

    函数计算如下：

    .. math::

        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    **输入：**
    
    - **input_x** (Tensor) - shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，数据类型支持float16或float32。

    **输出：**
    
    Tensor，与 `input_x` 的shape和数据类型相同。

    **异常：**
    
    - **TypeError** - `input_x` 不是Tensor。
    - **TypeError** - `input_x` 的数据类型非float16或float32。
