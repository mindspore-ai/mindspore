mindspore.Tensor.ptp
====================

.. py:method:: mindspore.Tensor.ptp(axis=None, keepdims=False)

    该函数名称是"peak to peak"的缩写。计算沿着axis的最大值与最小值的差值。

    .. note::
        不支持NumPy参数 `dtype` 和 `out` 。

    参数：
        - **axis** (Union[None, int, tuple(int)]) - 轴，在轴方向上可以计算范围。默认计算扁平数组的方差。默认值：None。
        - **keepdims** (bool) - 如果设为True，被删去的维度保留在结果中，且维度大小设为1。有了这个选项，结果将针对输入数组正确传递。默认值为False。

    返回：
        Tensor。

    异常：
        - **TypeError** - `self` 不是Tensor，或者 `axis` 和 `keepdims` 具有前面未指定的类型。