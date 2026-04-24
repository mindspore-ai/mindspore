mindspore.numpy.maximum
=======================

.. py:function:: mindspore.numpy.maximum(x1, x2, dtype=None)

    逐元素比较两个数组，返回每对数组元素中的最大值。

    比较两个数组，并返回一个包含逐元素最大值的新数组。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。 在Ascend上，不支持包含inf或NaN的输入数组。

    参数：
        - **x1** (Tensor) - 输入数组。
        - **x2** (Tensor) - 包含要比较元素的数组。 如果 ``x1.shape != x2.shape`` ，它们必须可以广播到一个共同的shape(即输出的shape)。
        - **dtype** (mindspore.dtype, 可选) - 默认值: ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，逐元素返回 `x1` 和 `x2` 中的最大值。如果 `x1` 和 `x2` 都是标量，返回标量。