mindspore.ops.unbind
========================

.. py:function:: mindspore.ops.unbind(input, dim=0)

    根据指定轴对输入矩阵进行分解。

    若输入Tensor在指定的轴上的rank为 `R` ，则输出Tensor的rank为 `(R-1)` 。

    给定一个shape为 :math:`(input_1, input_2, ..., input_R)` 的Tensor。如果存在 :math:`0 \le axis` ，则输出Tensor的shape为 :math:`(input_1, input_2, ..., input_{axis}, input_{axis+2}, ..., input_R)` 。

    参数：
        - **input** (Tensor) - 输入Tensor，其shape为 :math:`(input_1, input_2, ..., input_R)` 。rank必须大于0。
        - **dim** (int) - 指定矩阵分解的轴。取值范围为[-R, R)，默认值：0。

    返回：
        Tensor对象组成的tuple。每个Tensor对象的shape相同。

    异常：
        - **ValueError** - `dim` 超出[-R, R)范围。
