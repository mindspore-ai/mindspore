mindspore.numpy.cov
===================

.. py:function:: mindspore.numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None)

    给定数据和权重，估算一个协方差矩阵。

    协方差表示两个变量共同变化的程度。 如果以N维样本为例， :math:`X = [x_1, x_2, .., x_N]^T` ，则协方差矩阵元素 :math:`C_{ij}` 是 :math:`x_i` 和 :math:`x_j` 的协方差。 元素 :math:`C_{ii}` 是 :math:`x_i` 的方差。

    .. note::
        `fweights` 和 `aweights` 必须都为正，在NumPy中，如果检测到负值，将引发ValueError。 在MindSpore中，所有值将转化为正值。

    参数：
        - **m** (Union[Tensor, list, tuple]) - 一个1-D或2-D的Tensor，包含多个变量和观测值。 `m` 的每一行代表一个变量，每一列代表所有变量的某一观测值。 另请见下述 `rowvar` 。
        - **y** (Union[Tensor, list, tuple], 可选) - 一个附加的变量和观测值的集合。 `y` 的形式与 `m` 相同，默认值： ``None``。
        - **rowvar** (bool, 可选) - 如果 `rowvar` 为 ``True`` (默认值)，则每行代表一个变量，每列代表一个观测值。 否则，关系被转置：每列代表一个变量，每行代表一个观测值。默认值： ``True``。
        - **bias** (bool, 可选) - 默认的归一化( ``False`` )是通过 :math:`N - 1` 计算的， 其中 :math:`N` 是观测值的数量。 如果 `bias` 为 ``True`` ，则归一化通过 :math:`N` 计算。 这些值可通过关键词 `ddof` 被覆盖。
        - **ddof** (int, 可选) - 如果不为 ``None``，由 `bias` 设定的默认值将被覆盖。 请注意，即使指定了 `fweights` 和 `aweights`， :math:`ddof = 1` 也将返回无偏估计 ， :math:`ddof = 0` 将返回简单平均值。默认值： ``None`` 。
        - **fweights** (Union[Tensor, list, tuple], 可选) - 一个1-D的整数频率权重Tensor。 表示每个观测向量应该重复的次数。 默认值： ``None`` 。
        - **aweights** (Union[Tensor, list, tuple], 可选) - 一个1-D的观测向量权重Tensor。 被认为更重要的观测值的权重相对较大，被认为不那么重要的观测值的权重相对较小。如果 :math:`ddof = 0` ，权重Tensor可用于给观测向量分配概率。默认值： ``None``。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 结果的数据类型。 默认情况下，返回的数据类型将具有mstype.float32精度。 默认值： ``None`` 。

    返回：
        Tensor，变量的协方差矩阵。

    异常：
        - **TypeError** - 如果输入的类型不是上述指定类型。
        - **ValueError** - 如果 `m` 和 `y` 的维数错误。
        - **RuntimeError** - 如果 `aweights` 和 `fweights` 的维数大于2。