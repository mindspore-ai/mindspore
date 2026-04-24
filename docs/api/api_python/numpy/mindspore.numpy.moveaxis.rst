mindspore.numpy.moveaxis
=================================

.. py:function:: mindspore.numpy.moveaxis(a, source, destination)

    将数组的轴移动到新位置。
    其他轴保持原始顺序不变。

    参数：
        - **a** (Tensor) - 原数组，返回数组的shape和数据类型与 `a` 相同。
        - **source** (int, ints的序列) - 要移动的轴的原始位置。这些位置必须唯一。
        - **destination** (int, ints的序列) - 每个原始轴的新位置。这些位置也必须唯一。

    返回：
        Tensor，已经移动过轴的数组。

    异常：
        - **ValueError** - 如果轴超出范围 :math:`[-a.ndim, a.ndim)` ，或者轴中包含重复项。