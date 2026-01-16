mindspore.ops.inverse
=====================

.. py:function:: mindspore.ops.inverse(input)

    计算输入矩阵的逆。

    .. note::
        `input` 至少是两维的，最后两个维度大小相同，并且矩阵需要可逆。不支持复数类型的输入。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
