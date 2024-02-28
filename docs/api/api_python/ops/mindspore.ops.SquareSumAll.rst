mindspore.ops.SquareSumAll
============================

.. py:class:: mindspore.ops.SquareSumAll

    计算输入Tensor的平方和。

    .. math::
        \left\{\begin{matrix}out_{x} = {\textstyle \sum_{0}^{N}} (x_{i})^2
        \\out_{y} = {\textstyle \sum_{0}^{N}} (y_{i})^2
        \end{matrix}\right.
    
    .. note::
        SquareSumAll只支持float16和float32类型的输入值。

    输入：
        - **x** (Tensor) - SquareSumAll的输入，其数据类型为数值型，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **y** (Tensor) - 数据类型和shape与 `x` 相同。

    输出：
        - **output_x** (Tensor) - 数据类型与 `x` 相同。
        - **output_y** (Tensor) - 数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 或 `y` 不是Tensor。
        - **ValueError** - 如果 `x` 与 `y` 的shape不一致。
