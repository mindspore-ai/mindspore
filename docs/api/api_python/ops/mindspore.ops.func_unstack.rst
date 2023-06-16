mindspore.ops.unstack
=======================

.. py:function:: mindspore.ops.unstack(input_x, axis=0)

    根据指定轴对输入矩阵进行分解，与 :func:`mindspore.ops.stack` 函数操作相反。
    若输入Tensor的rank为 `R` ，则输出Tensor的rank为 `(R-1)` 。

    参数：
        - **input_x** (Tensor) - 输入Tensor，其shape为 :math:`(x_1, x_2, ..., x_R)` 。rank必须大于0。
        - **axis** (int) - 指定矩阵分解的轴。取值范围为[-R,R)，默认值： ``0`` 。

    返回：
        Tensor对象组成的tuple。每个Tensor对象的shape相同。
        给定一个shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。如果存在 :math:`0 \le axis` ，则输出Tensor的shape为 :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)` 。

    异常：
        - **ValueError** - axis超出[-len(input_x.shape), len(input_x.shape))范围。
