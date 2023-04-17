mindspore.Tensor.var
====================

.. py:method:: mindspore.Tensor.var(axis=None, ddof=0, keepdims=False)

    在指定维度上的方差。

    方差是平均值的平方偏差的平均值，即：:math:`var = mean(abs(x - x.mean())**2)` 。

    返回方差值，默认情况下计算展开Tensor的方差，否则在指定维度上计算。

    .. note::
        不支持NumPy参数 `dtype` 、 `out` 和 `where` 。

    参数：
        - **axis** (Union[None, int, tuple(int)]) - 维度，在指定维度上计算方差。其默认值是展开Tensor的方差。默认值： ``None`` 。
        - **ddof** (int) - δ自由度。默认值： ``0`` 。计算中使用的除数是 :math:`N - ddof` ，其中 :math:`N` 表示元素的数量。
        - **keepdims** (bool) - 默认值： ``False`` 。

    返回：
        含有方差值的Tensor。