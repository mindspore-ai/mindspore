mindspore.ops.Unstack
=======================

.. py:class:: mindspore.ops.Unstack(axis=0)

    根据指定轴对输入矩阵进行分解。

    在指定轴上，矩阵分解的rank为 `R` 的Tensor，输出rank为 `(R-1)` 的Tensor。

    给定一个shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。如果存在 :math:`0 \le axis` ，则输出Tensor的shape为 :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)` 。

    与Stack函数操作相反。

    **参数：**

    - **axis** (int) - 指定矩阵分解的轴。取值范围为[-R,R)，默认值：0。

    **输入：**

    - **input_x** (Tensor) - 输入Tensor，其shape为 :math:`(x_1, x_2, ..., x_R)` 。rank必须大于0。

    **输出：**

    Tensor对象组成的tuple。每个Tensor对象的shape相同。

    **异常：**

    - **ValueError** - axis超出[-len(input_x.shape), len(input_x.shape))范围。