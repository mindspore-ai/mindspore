mindspore.ops.Concat
======================

.. py:class:: mindspore.ops.Concat(axis=0)

    在指定轴上拼接输入Tensor。

    更多参考详见 :func:`mindspore.ops.concat`。

    参数：
        - **axis** (int，可选) - 表示指定的轴。默认值： ``0`` 。

    输入：
        - **input_x** (Union[tuple, list]) - 输入为Tensor组成的tuple或list。假设在这个tuple或list中有两个Tensor，即x1和x2。要在0轴方向上执行 `Concat` ，除0轴外，其他轴的shape都应相等，即 :math:`x1.shape[1] == x2.shape[1], x1.shape[2] == x2.shape[2], ..., x1.shape[R] == x2.shape[R]` ，其中 :math:`R` 表示最后一个轴。

    输出：
        Tensor，shape为 :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)` 。数据类型与 `input_x` 相同。
