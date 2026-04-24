mindspore.numpy.minimum
=======================

.. py:function:: mindspore.numpy.minimum(x1, x2, dtype=None)

    逐元素比较两个数组，返回每对数组元素中的最小值。

    比较两个数组，并返回一个包含逐元素最小值的新数组。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。 在Ascend上，不支持包含inf或NaN的输入数组。

    参数：
        - **x1** (Tensor) - 第一个需比较的输入数组。
        - **x2** (Tensor) - 第二个需比较的输入数组。
        - **dtype** (mindspore.dtype, 可选) - 默认值: ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，逐元素返回 `x1` 和 `x2` 中的最小值。

    异常：
        - **TypeError** - 如果输入类型不是上述指定类型。
        - **ValueError** - 如果 `x1` 和 `x2` 不能被广播。