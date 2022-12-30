mindspore.ops.FillDiagonal
==========================

.. py:class:: mindspore.ops.FillDiagonal(fill_value, wrap=False)

    填充至少具有二维的Tensor的主对角线。当 `dims>2` 时，输入的所有维度必须等长。此函数就地修改输入Tensor，并返回输入Tensor。

    参数：
        - **fill_value** (float) - 填充值。
        - **wrap** (bool，可选) - 如果设置为True, 表示高矩阵N列之后的对角线 `wrapped` ，否则不处理，默认值：False。

    输入：
        - **input_x** (Tensor) - shape为 :math:`(x_1, x_2, ..., x_R)` ，其数据类型必须为：float32、int32或者int64。

    输出：
        - **y** (Tensor) - Tensor，和输入 `x` 具有相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `input_x` 的dtype不是：float32、int32或者int64。
        - **ValueError** - 如果 `input_x` 的维度没有大于1。
        - **ValueError** - 当维度大于2时，每个轴的大小不相等。
