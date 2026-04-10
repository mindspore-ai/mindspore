mindspore.numpy.rollaxis
=================================

.. py:function:: mindspore.numpy.rollaxis(x, axis, start=0)

    将指定轴向后滚动，直到它位于给定的位置。其他轴的相对位置保持不变。

    参数：
        - **x** (Tensor) - 要进行调换的Tensor。
        - **axis** (int) - 要滚动的轴。
        - **start** (int) - 默认值: ``0`` 。如果 :math:`start \leq axis` ，则将轴向后滚动，直到它位于这个位置( ``start`` )。如果 :math:`start > axis` ，则将轴滚动直到它位于此位置之前( ``start`` )。如果 :math:`start < 0` ，则 ``start`` 会被归一化为非负数（详细信息见源码）。

    返回：
        转置后的Tensor，数据类型与原Tensor `x` 相同。

    异常：
        - **TypeError** - 如果 ``axis`` 或 ``start`` 不是整数，或者 ``x`` 不是Tensor。
        - **ValueError** - 如果 ``axis`` 不在 :math:`[-ndim, ndim-1]` 范围内，或者 ``start`` 不在 :math:`[-ndim, ndim]` 范围内。