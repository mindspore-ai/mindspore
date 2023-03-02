mindspore.ops.FillDiagonal
==========================

.. py:class:: mindspore.ops.FillDiagonal(fill_value, wrap=False)

    填充至少具有二维的Tensor的主对角线。当输入的维度大于2时，所有维度必须等长。此函数原地修改输入Tensor，并返回输入Tensor。

    参数：
        - **fill_value** (float) - `input_x` 对角线的填充值。
        - **wrap** (bool，可选) - 控制对于一个高矩阵（即矩阵的行数大于列数），对角线元素是否继续延伸到剩余的行。具体效果详见下方代码样例。默认值：False。

    输入：
        - **input_x** (Tensor) - shape为 :math:`(x_1, x_2, ..., x_R)` ，其数据类型必须为：float32、int32或者int64。

    输出：
        - **y** (Tensor) - Tensor，和输入 `input_x` 具有相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `input_x` 的dtype不是：float32、int32或者int64。
        - **ValueError** - 如果 `input_x` 的维度没有大于1。
        - **ValueError** - 当维度大于2时，每个轴的大小不相等。
