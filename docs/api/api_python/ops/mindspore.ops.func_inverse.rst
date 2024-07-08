mindspore.ops.inverse
=====================

.. py:function:: mindspore.ops.inverse(input)

    计算输入矩阵的逆。

    参数：
        - **input** (Tensor) - 计算的矩阵。`input` 至少是两维的，最后两个维度大小相同，并且矩阵需要可逆。

    返回：
        Tensor，其类型和shape与 `input` 的类型和shape相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 最后两个维度的大小不相同。
        - **ValueError** - `input` 的维数小于2。
