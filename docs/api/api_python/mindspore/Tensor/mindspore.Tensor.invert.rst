mindspore.Tensor.invert
=======================

.. py:method:: mindspore.Tensor.invert()

    按位翻转当前Tensor。

    .. math::
        out_i = \sim x_{i}

    其中 `x` 表示当前Tensor。

    返回：
        Tensor，shape和类型与当前Tensor相同。

    异常：
        - **TypeError** - 当前Tensor的数据类型不为int16或uint16。