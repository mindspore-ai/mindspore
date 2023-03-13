mindspore.ops.t
===============

.. py:function:: mindspore.ops.t(input)

    转置二维Tensor。一维Tensor按原样返回。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，`input` 的转置。

    异常：
        - **ValueError** - `input` 的维度大于2。
