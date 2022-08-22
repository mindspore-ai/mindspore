mindspore.Tensor.inv
====================

.. py:method:: mindspore.Tensor.inv()

    逐元素计算当前Tensor的倒数。

    .. math::
        out_i = \frac{1}{x_{i} }

    其中 `x` 表示当前Tensor。

    返回：
        Tensor，shape和类型与当前Tensor相同。

    异常：
        - **TypeError** - 当前Tensor的数据类型不为float16、float32或int32。