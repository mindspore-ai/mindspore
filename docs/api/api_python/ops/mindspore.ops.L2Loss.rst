mindspore.ops.L2Loss
====================

.. py:class:: mindspore.ops.L2Loss

    用于计算L2范数的一半，但不对结果进行开方操作。

    把输入设为x，输出设为loss。

    .. math::
        loss = \frac{\sum x ^ 2}{2}

    输入：
        - **input_x** (Tensor) - 用于计算L2范数的Tensor。数据类型必须为float16或float32。

    输出：
        Tensor，具有与 `input_x` 相同的数据类型的Scalar Tensor。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `input_x` 的数据类型既不是float16也不是float32。
