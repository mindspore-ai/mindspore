mindspore.numpy.var
===================

.. py:function:: mindspore.numpy.var(x, axis=None, ddof=0, keepdims=False)

    计算沿指定轴的方差。 方差是各值与平均值的平方差的平均数。

    默认情况下，返回计算展平数组的方差，否则沿指定轴计算方差。

    .. note::
        不支持NumPy参数 `dtype` 、 `out` 、 `where` 。

    参数：
        - **x** (Tensor) - 要进行计算的Tensor。
        - **axis** (Union[None, int, tuple(int)]) - 计算方差所沿的单个或多个轴。取默认值 `None` 时，在展平数组上计算。
        - **ddof** (int) - 自由度偏差。默认值： ``0`` 。 计算中使用的除数是 :math:`(N - ddof)` 。其中 :math:`N` 代表元素的数量。
        - **keepdims** (bool) - 如果设置为 `True` ，减少的轴在结果中保留为大小为1的维度。若使用此选项，结果会广播到和输入Tensor同一个维度数。如果传入默认值，则 `keepdims` 参数不会传递到Tensor子类的var方法中，而任何非默认值将会传递。如果子类中方法未实现 `keepdims` ，则会引发异常。默认值： ``False`` 。

    返回：
        Tensor。
