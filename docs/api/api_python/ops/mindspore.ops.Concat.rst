mindspore.ops.Concat
=====================

.. py:class:: mindspore.ops.Concat(axis=0)

    在指定轴上拼接输入Tensor。

    输入的是一个tuple。其元素秩相同，即 `R` 。将给定的轴设为 `m` ，并且 :math:`0 \le m < R` 。数量设为 `N` 。对于第 :math:`i` 个数据， :math:`t_i` 的shape为 `(x_1, x_2, ..., x_{mi}, ..., x_R)` 。 :math:`x_{mi}` 是第 :math:`i` 个元素的第 :math:`m` 个维度。然后，输出tensor的shape为：

    .. math::
        (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    .. note::
        "axis"的取值范围为[-dims, dims - 1]。"dims"为"input_x"的维度长度。

    **参数：**

    - **axis** (int) - 表示指定的轴。默认值：0。

    **输入：**

    - **input_x** (tuple, list) - 输入为Tensor tuple或Tensor list。假设在这个tuple或list中有两个Tensor，即x1和x2。要在0轴方向上执行 `Concat` ，除0轴外，所有其他轴都应相等，即 :math:`x1.shape[1] == x2.shape[1]，x1.shape[2] == x2.shape[2]，...，x1.shape[R] == x2.shape[R]` ，其中 :math:`R` 表示最后一个轴。

    **输出：**

    Tensor，shape为 :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)` 。数据类型与 `input_x` 相同。

    **异常：**

    - **TypeError** - `axis` 不是int。