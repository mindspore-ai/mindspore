mindspore.ops.Geqrf
===================

.. py:class:: mindspore.ops.Geqrf

    用于计算QR分解的低级函数。此函数返回两个Tensor（y，tau）。

    计算 `x` 的QR分解。 `Q` 和 `R` 矩阵都存储在同一个输出Tensor `y` 中。
    
    `R` 的元素存储在对角线及上方。隐式定义矩阵 `Q` 的基本反射器（或户主向量）存储在对角线下方。

    输入：
        - **x** (Tensor) - shape为 :math:`(m, n)` ，输入必须为两维矩阵，dtype为float32、float64。

    输出：
        - **y** (Tensor) - shape为 :math:`(m, n)` ，与 `x` 具有相同的dtype。
        - **tau** (Tensor) - shape为 :math:`(p,)` ，并且 :math:`p = min(m, n)` ，与 `x` 具有相同的dtype。

    异常：
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **TypeError** - 如果 `x` 的dtype不是float32、float64中的一个。
        - **ValueError** - 如果 `x` 的维度不等于2。
