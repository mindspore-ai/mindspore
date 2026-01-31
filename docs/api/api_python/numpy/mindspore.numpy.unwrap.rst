mindspore.numpy.unwrap
======================

.. py:function:: mindspore.numpy.unwrap(p, discont=3.141592653589793, axis=-1)

    通过加或减 ``2*pi`` ，改变数组相邻元素的差值实现解卷绕。 对于弧度数组 `p` ，将其大于 `discont` 的相邻元素之差的绝对值加或减 ``2*pi`` ，直到相邻元素之差的绝对值小于 `discont` ，沿指定轴执行。

    .. note::
        对于相邻元素之差的绝对值与pi的差值很小时，由于舍入误差的差异，解卷绕可能会与NumPy不同。

    参数：
        - **p** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。
        - **discont** (float, 可选) - 值之间的最大不连续性，默认值： ``pi`` 。
        - **axis** (int, 可选) - 解卷绕操作所沿的轴。 默认值： ``-1`` 。

    返回：
        Tensor。

    异常：
        - **ValueError** - 如果轴超出范围。