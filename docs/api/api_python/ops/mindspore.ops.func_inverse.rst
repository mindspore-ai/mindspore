mindspore.ops.inverse
=====================

.. py:function:: mindspore.ops.inverse(x)

    计算输入矩阵的逆。

    参数：
        - **x** (Tensor) - 计算的矩阵。`x` 至少是两维的，最后两个维度大小相同。

    返回：
        Tensor，shape和类型和 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `x` 最后两个维度的大小不相同。
        - **ValueError** - `x` 的维数小于2。
