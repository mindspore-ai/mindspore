mindspore.numpy.corrcoef
========================

.. py:function:: mindspore.numpy.corrcoef(x, y=None, rowvar=True, dtype=None)

    返回皮尔逊积矩相关系数。
    请参考有关协方差(cov)的文档以获取更多详情。 相关系数矩阵R和协方差矩阵C的关系表示为 :math:`R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} * C_{jj} } }` 。 R的值介于-1和1之间，包括-1和1。

    .. note::
        目前不支持复数。

    参数：
        - **x** (Union[int, float, bool, tuple, list, Tensor]) - 这是一个1-D或2-D的数组，包含多个变量和观测值。 每一行的 `x` 代表一个变量，每一列代表所有这些变量的一个观测值。 详细见下文 `rowvar` 的说明。
        - **y** (Union[int, float, bool, tuple, list, Tensor], 可选) - 这是一个附加的变量和观测值集合，默认值： ``None`` 。
        - **rowvar** (bool, 可选) - 如果 `rowvar` 是 ``True`` (默认)，则每一行代表一个变量，列中包含观测值。 否则，关系被转置：每一列代表一个变量，行包含观测值。 默认值： ``True`` 。
        - **dtype**  (mindspore.dtype, 可选) - 结果的数据类型。 默认情况下，返回的数据类型至少为float32精度。 默认值： ``None`` 。

    返回：
        Tensor。 变量的相关系数矩阵。

    异常：
        - **TypeError** - 如果输入的类型不符合上述规定。
        - **ValueError** - 如果 `x` 和 `y` 的维数不正确。