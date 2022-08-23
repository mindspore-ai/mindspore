mindspore.Tensor.log1p
======================

.. py:method:: mindspore.Tensor.log1p()

    对当前Tensor逐元素加一后计算自然对数。

    其中 `x` 表示当前Tensor。

    .. math::
        out_i = {log_e}(x_i + 1)

    返回：
        Tensor，与 `x` 的shape相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型非float16或float32。