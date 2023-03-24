mindspore.ops.mvlgamma
=======================

.. py:function:: mindspore.ops.mvlgamma(input, p)

    逐元素计算 `p` 维多元对数伽马函数值。

    Mvlgamma计算公式如下：

    .. math::

        \log (\Gamma_{p}(input))=C+\sum_{i=1}^{p} \log (\Gamma(input-\frac{i-1}{2}))
    
    其中 :math:`C = \log(\pi) \times \frac{p(p-1)}{4}` ，:math:`\Gamma(\cdot)` 为Gamma函数。

    参数：
        - **input** (Tensor) - 多元对数伽马函数的输入Tensor，支持数据类型为float32和float64。其shape为 :math:`(N,*)` ，其中 :math:`*` 为任意数量的额外维度。 `input` 中每个元素的值必须大于 :math:`(p - 1) / 2` 。
        - **p** (int) - 进行计算的维度，必须大于等于1。

    返回：
        Tensor。shape和类型与 `input` 一致。

    异常：
        - **TypeError** - `input` 的数据类型不是float32或者float64。
        - **TypeError** - `p` 不是int类型。
        - **ValueError** - `p` 小于1。
        - **ValueError** - `input` 中不是所有元素的值都大于 :math:`(p - 1) / 2` 。
