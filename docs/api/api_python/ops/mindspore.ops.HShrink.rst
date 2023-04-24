mindspore.ops.HShrink
=====================

.. py:class:: mindspore.ops.HShrink(lambd=0.5)

    Hard Shrink激活函数。

    更多参考详见 :func:`mindspore.ops.hardshrink`。

    参数：
        - **lambd** (float，可选) - Hard Shrink公式定义的阈值 :math:`\lambda` 。默认值： ``0.5`` 。

    输入：
        - **input_x** (Tensor) - Hard Shrink的输入，数据类型为float16或float32。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。
