mindspore.Tensor.std
====================

.. py:method:: mindspore.Tensor.std(axis=None, ddof=0, keepdims=False)

    计算指定维度的标准差。
    标准差是方差的算术平方根，如：:math:`std = sqrt(mean(abs(x - x.mean())**2))` 。

    返回标准差，默认情况下计算展开数组的标准差，否则在指定维度上计算。

    .. note::
        不支持NumPy参数 `dtype` 、 `out` 和 `where` 。

    参数：
        - **axis** (Union[None, int, tuple(int)]) - 在该维度上计算标准差。默认值：`None` 。如果为 `None` ，则计算展开数组的标准偏差。
        - **ddof** (int) - δ自由度。计算中使用的除数是 :math:`N - ddof` ，其中 :math:`N` 表示元素的数量。默认值：0。
        - **keepdims** - 默认值：`False`。

    返回：
        含有标准差数值的Tensor。