mindspore.ops.Erf
=================

.. py:class:: mindspore.ops.Erf

    逐元素计算 `x` 的高斯误差函数。

    更多参考详见 :func:`mindspore.ops.erf`。

    输入：
        - **x** (Tensor) - 高斯误差函数的输入Tensor。支持数据类型：

          - Ascend： float16、float32。
          - GPU/CPU： float16、float32、float64。

    输出：
        Tensor，具有与 `x` 相同的数据类型和shape。
