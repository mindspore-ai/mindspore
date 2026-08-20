mindspore.mint.logaddexp
========================

.. py:function:: mindspore.mint.logaddexp(input, other)

    计算输入的指数和的对数。
    该函数可以很好地解决统计学中的一个问题，即计算得到的事件概率可能非常小，以至于超出正常浮点数的范围。

    .. math::

        out_i = \log(\exp(input_i) + \exp(other_i))

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。其数据类型必须是float。
        - **other** (Tensor) - 输入tensor。其数据类型必须是float。如果 `input` 的shape不等于 `other` 的shape，它们必须被广播成相同shape。

    返回：
        Tensor，数据类型与输入 `input` 和 `other` 相同。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor。
        - **TypeError** - `input` 或 `other` 的数据类型不是float。
