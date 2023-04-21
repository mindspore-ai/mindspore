mindspore.ops.batch_dot
=======================

.. py:function:: mindspore.ops.batch_dot(x1, x2, axes=None)

    当输入的两个Tensor是批量数据时，对其进行批量点积操作。

    .. math::
        output = x1[batch, :] * x2[batch, :]

    参数：
        - **x1** (Tensor) - 第一个输入Tensor，数据类型为float32且 `x1` 的秩必须大于或等于2。
        - **x2** (Tensor) - 第二个输入Tensor，数据类型为float32。 `x2` 的数据类型应与 `x1` 相同，`x2` 的秩必须大于或等于2。
        - **axes** (Union[int, tuple(int), list(int)]) - 指定为单值或长度为2的tuple和list，分别指定 `a` 和 `b` 的维度。如果传递了单个值 `N`，则自动从输入 `a` 的shape中获取最后N个维度，从输入 `b` 的shape中获取最后N个维度，分别作为每个维度的轴。默认值： ``None`` 。

    返回：
        Tensor， `x1` 和 `x2` 的批量点积。例如：输入 `x1` 的shape为 :math:`(batch, d1, axes, d2)`，`x2` shape为 :math:`(batch, d3, axes, d4)`，则输出shape为 :math:`(batch, d1, d2, d3, d4)`，其中d1和d2表示任意数字。

    异常：
        - **TypeError** - `x1` 和 `x2` 的类型不相同。
        - **TypeError** - `x1` 或 `x2` 的数据类型不是float32。
        - **ValueError** - `x1` 或 `x2` 的秩小于2。
        - **ValueError** - 在轴中使用了代表批量的维度。
        - **ValueError** - `axes` 的长度小于2。
        - **ValueError** - `axes` 不是其一：None，int，或(int, int)。
        - **ValueError** - 如果 `axes` 为负值，低于输入数组的维度。
        - **ValueError** - 如果 `axes` 的值高于输入数组的维度。
        - **ValueError** - `x1` 和 `x2` 的批处理大小不相同。
