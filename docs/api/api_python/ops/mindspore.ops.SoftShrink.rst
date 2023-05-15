mindspore.ops.SoftShrink
========================

.. py:class:: mindspore.ops.SoftShrink(lambd=0.5)

    Soft Shrink激活函数。

    更多参考详见 :func:`mindspore.ops.softshrink`。

    参数：
        - **lambd** (float，可选) - :math:`\lambda` ，应大于等于0。默认值： ``0.5`` 。

    输入：
        - **input_x** (Tensor) - Soft Shrink的输入，数据类型为float16或float32。

    输出：
        Tensor，shape和数据类型与 `input_x` 相同。
