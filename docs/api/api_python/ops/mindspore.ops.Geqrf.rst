mindspore.ops.Geqrf
===================

.. py:class:: mindspore.ops.Geqrf

    将矩阵分解为正交矩阵 `Q` 和上三角矩阵 `R` 的乘积。该过程称为QR分解： :math:`A = QR` 。

    `Q` 和 `R` 矩阵都存储在同一个输出Tensor `y` 中。 `R` 的元素存储在对角线及上方。隐式定义矩阵 `Q` 的基本反射器（或户主向量）存储在对角线下方。

    此函数返回两个Tensor（ `y`, `tau` ）。

    输入：
        - **x** (Tensor) - shape为 :math:`(*, m, n)` ，输入矩阵维度必须为大于等于两维，支持dtype为float32、float64、complex64、complex128。

    输出：
        - **y** (Tensor) - shape为 :math:`(*, m, n)` ，与 `x` 具有相同的dtype。
        - **tau** (Tensor) - shape为 :math:`(*, p)` ，并且 :math:`p = min(m, n)` ，与 `x` 具有相同的dtype。

    异常：
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **TypeError** - 如果 `x` 的dtype不是float32、float64、complex64、complex128中的一个。
        - **ValueError** - 如果 `x` 的维度小于2。
