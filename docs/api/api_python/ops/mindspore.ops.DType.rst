mindspore.ops.DType
====================

.. py:class:: mindspore.ops.DType

    计算输入Tensor的数据类型，且返回的数据类型为 `mindspore.dtype` 。

    输入：
        - **input_x** (Tensor) - 输入Tensor，其shape为 :math:`(x1, x2, ..., xR)` 。

    输出：
        `mindspore.dtype` ，输入Tensor的数据类型。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。

