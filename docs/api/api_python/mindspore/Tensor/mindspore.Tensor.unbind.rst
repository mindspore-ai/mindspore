mindspore.Tensor.unbind
========================

.. py:method:: mindspore.Tensor.unbind(dim=0)

    根据指定轴对输入矩阵进行分解。

    若输入Tensor在指定的轴上的rank为 `R` ，则输出Tensor的rank为 `(R-1)` 。

    给定一个shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。如果存在 :math:`0 \le axis` ，则输出Tensor的shape为 :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)` 。

    参数：
        - **dim** (int) - 指定矩阵分解的轴。取值范围为[-R, R)，默认值：0。

    返回：
        Tensor对象组成的tuple。每个Tensor对象的shape相同。

    异常：
        - **ValueError** - `dim` 超出[-R, R)范围。
