mindspore.ops.Split
====================

.. py:class:: mindspore.ops.Split(axis=0, output_num=1)

    根据指定的轴和分割数量对输入Tensor进行分割。

    更多参考详见 :func:`mindspore.ops.split`。

    参数：
        - **axis** (int) - 指定分割轴。默认值：0。
        - **output_num** (int) - 指定分割数量。其值为正整数。默认值：1。

    输入：
        - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。

    输出：
        tuple[Tensor]，每个输出Tensor的shape相同，即 :math:`(y_1, y_2, ..., y_S)` 。数据类型与 `input_x` 的相同。

