mindspore.numpy.std
===================

.. py:function:: mindspore.numpy.std(x, axis=None, ddof=0, keepdims=False)

    沿指定轴计算标准差。标准差是平方偏差平均值的平方根，即 :math:`std = sqrt(mean(abs(x - x.mean())**2))` 。

    返回标准差，默认计算展平数组的标准差，否则在指定轴上计算。

    .. note::
        不支持NumPy参数 `dtype` 、 `out` 、 `where` 。

    参数：
        - **x** (Tensor) - 进行计算的Tensor。
        - **axis** (Union[None, int, tuple(int)]) - 计算标准差所沿的单个或多个轴。 默认值： ``None`` 。如果为 `None` ，计算展平数组的标准差。
        - **ddof** (int) - 自由度偏差。 计算中使用的除数为 :math:`N - ddof` ，其中 :math:`N` 表示元素数量。 默认值： ``0``。
        - **keepdims** - 如果设置为 ``True`` ，减少的轴在结果中保留为大小为1的维度。 若使用此选项，结果会广播到与输入Tensor同一个维度数。 如果传入默认值，则 `keepdims` 参数不会传递到Tensor子类的std方法中，而任何非默认值将会传递。 如果子类中方法未实现 `keepdims` ，则会引发异常。默认值： ``False`` 。

    返回：
        标准差Tensor。