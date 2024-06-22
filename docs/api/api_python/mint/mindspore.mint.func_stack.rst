mindspore.mint.stack
======================

.. py:function:: mindspore.mint.stack(tensors, dim=0)

    在指定轴上对输入Tensor序列进行堆叠。

    输入秩为 `R` 的Tensor序列，则输出秩为 `(R+1)` 的Tensor。

    给定输入Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。若输入Tensor的长度为 `N` 。如果存在 :math:`dim \ge 0` ，则输出Tensor的shape为 :math:`(x_1, x_2, ..., x_{dim}, N, x_{dim+1}, ..., x_R)` 。

    参数：
        - **tensors** (Union[tuple, list]) - 输入多个Tensor对象组成的tuple或list，每个Tensor具有相同shape和数据类型。
        - **dim** (int) - 指定堆叠运算的轴。取值范围为[-(R+1), R+1)。默认值： ``0`` 。

    返回：
        堆叠运算后的Tensor，数据类型和 `tensors` 的相同。

    异常：
        - **TypeError** - `tensors` 中元素的数据类型不相同。
        - **ValueError** - `dim` 不在[-(R+1),R+1)范围中，或 `tensors` 中元素的shape不相同。
