mindspore.Tensor.atanh
======================

.. py:method:: mindspore.Tensor.atanh()

    逐元素计算输入张量的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(x_{i})

    .. warning::
        这是一个实验性原型接口，可以更改或删除。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型既不是float16，也不是float32。