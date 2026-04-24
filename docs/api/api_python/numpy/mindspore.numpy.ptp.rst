mindspore.numpy.ptp
===================

.. py:function:: mindspore.numpy.ptp(x, axis=None, keepdims=False)

    沿某个轴的值范围(最大值 - 最小值)。该函数名称来自于“peak to peak”的首字母缩写。

    .. note::
        不支持NumPy参数 `dtype` 和 `out` 。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **axis** (Union[None, int, tuple(int)]) - 计算范围所沿的单个或多个轴。默认在计算展平的数组上计算。默认值： ``None`` 。
        - **keepdims** (bool) - 如果设置为 ``True`` ，减少的轴在结果中作为大小为1的维度保留。若使用此选项，结果会广播到和输入Tensor同一个维度数。如果传入默认值，则 `keepdims` 参数不会传递到Tensor子类的ptp方法中，而任何非默认值将会传递。默认值： ``False`` 。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入类型不是上述指定类型。
