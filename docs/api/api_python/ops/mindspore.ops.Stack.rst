mindspore.ops.Stack
====================

.. py:class:: mindspore.ops.Stack(axis=0)

    根据指定轴对输入Tensor序列进行堆叠运算。

    输入秩为 `R` 的Tensor序列，且输出秩为 `(R+1)` 的Tensor。

    给定输入Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。将输入Tensor的数量设为 `N` 。如果存在 :math:`0 \le axis` ，则输出Tensor的shape为 :math:`(x_1, x_2, ..., x_{axis}, N, x_{axis+1}, ..., x_R)` 。

    **参数：**

    - **axis** (int) - 指定堆叠运算的轴。取值范围为[-(R+1), R+1)。默认值：0。

    **输入：**

    - **input_x** (Union[tuple, list]) - 输入多个Tensor对象组成的tuple或list，每个Tensor具有相同shape和数据类型。

    **输出：**

    堆叠运算后的Tensor，数据类型和 `input_x` 的相同。

    **异常：**

    - **TypeError** - `input_x` 中元素的数据类型不相同。
    - **ValueError** - `input_x` 的长度不大于1，或axis不在[-(R+1),R+1)范围中，或 `input_x` 中元素的shape不相同。