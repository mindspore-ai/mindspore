mindspore.ops.inverse
=====================

.. py:function:: mindspore.ops.inverse(input)

    计算输入矩阵的逆。

    参数：
        - **input** (Tensor) - 计算的矩阵。`input` 至少是两维的，最后两个维度大小相同。

    返回：
        Tensor，shape和类型和 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 最后两个维度的大小不相同。
        - **ValueError** - `input` 的维数小于2。
