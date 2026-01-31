mindspore.numpy.vander
=================================

.. py:function:: mindspore.numpy.vander(x, N=None, increasing=False)

    生成一个范德蒙德矩阵。
    输出矩阵的列是输入向量的幂。幂的顺序由 `increasing` boolean型参数决定。具体而言，当 `increasing` 为 ``False`` 时，第 `i` 列输出是按元素递增的输入向量，其幂为 :math:`N-i-1` 。这样的一个每行都有几何级数的矩阵被称为范德蒙德矩阵。

    参数：
        - **x** (Union[list, tuple, Tensor]) - 输入的一维数组。
        - **N** (int, 可选) - 输出结果的列数。如果未指定 `N` ，则返回一个 :math:`N=len(x)` 的方阵。
        - **increasing** (bool, 可选) - 列的幂次顺序。如果为 ``True`` ，则幂次从左到右递增，如果为 ``False`` ，则幂次反向，默认值： ``False`` 。

    返回：
        Tensor，范德蒙德矩阵，如果 `increasing` 为 ``False`` ，则第一列为 :math:`x^{(N-1)}` ，第二列为 :math:`x^{(N-2)}` ，依此类推。如果 `increasing` 为 ``True`` ，则列为 :math:`x^0, x^1, ..., x^{(N-1)}` 。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果输入的 `x` 不是一维，或 `N` 小于0。