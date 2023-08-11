mindspore.ops.Eps
=================

.. py:class:: mindspore.ops.Eps

    创建一个与输入数据类型和shape都相同的Tensor，元素值为对应数据类型能表达的最小值。

    更多参考详见 :func:`mindspore.ops.eps`。

    输入：
        - **x** (Tensor) - 用于获取其数据类型能表达的最小值的任意维度的Tensor。数据类型必须为float16、float32或者float64。

    输出：
        Tensor，具有与 `x` 相同的数据类型和shape，填充了 `x` 数据类型的最小值。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或者float64。
