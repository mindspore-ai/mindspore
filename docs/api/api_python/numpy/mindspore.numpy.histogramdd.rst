mindspore.numpy.histogramdd
===========================

.. py:function:: mindspore.numpy.histogramdd(sample, bins=10, range=None, weights=None, density=False)

    计算数据的多维直方图。

    .. note::
        不支持已弃用的NumPy参数 `normed` 。

    参数：
        - **sample** (Union[list, tuple, Tensor]) - 要进行直方图统计的数据，可以是 `(N, D)` 数组，或者 `(D, N)` 的类数组。 注意当使用类数组为样本的非常规解释：当样本数组时，每行是D维空间中的一个坐标，例如 ``histogramdd(np.array([p1, p2, p3]))`` 。 当样本为类数组时，每个元素是单一坐标系的值列表，如 ``histogramdd((X, Y, Z))`` 。 应优先考虑第一种形式。
        - **bins** (Union[int, tuple, list], 可选) - 桶的规格： 一个数组序列用于描述沿每个维度单调递增的桶的边界。 每个维度的桶的数量 ``(nx, ny, ... =bins)`` 。 所有维度的桶的数量 ``(nx=ny...=bins)`` 。默认值： `10` 。
        - **range** (Union[list, tuple], 可选) - 长度为D的序列，每个维度可以选择一个 ``(lower, upper)`` tuple来指定桶的外边界，如果边界在 `bins` 没有特别给出。 序列中的None项意味着其对应维度使用最小值和最大值。 默认值None相当于给 `D` 中Tuple传递None值。默认值： ``None`` 。
        - **weights** (Union[list, tuple, Tensor], 可选) - 一个shape为 ``(N,)`` 的数组，其中的值 ``w_i`` 表示每个样本的权重 ``(x_i, y_i, z_i, …)`` 。默认值： ``None`` 。
        - **density**  (boolean, 可选) - 如果为 False（默认值），则返回每个区间中的样本数量。如果为 ``True``，则返回该区间的概率密度函数 ``bin_count / sample_count / bin_volume`` 。 默认值： ``False`` 。

    返回：
        Tensor, 元素为Tensor的list， 直方图的值和桶边界。
