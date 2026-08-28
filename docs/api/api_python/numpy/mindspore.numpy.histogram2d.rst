mindspore.numpy.histogram2d
===========================

.. py:function:: mindspore.numpy.histogram2d(x, y, bins=10, range=None, weights=None, density=False)

    计算数据的二维直方图。

    .. note::
        不支持已弃用的NumPy参数 `normed` 。

    参数：
        - **x** (Union[list, tuple, Tensor]) - 一个shape为(N,)的数组，包含要进行直方图统计的点的x坐标。
        - **y** (Union[list, tuple, Tensor]) - 一个shape为(N,)的数组，包含要进行直方图统计的点的y坐标。
        - **bins** (Union[int, tuple, list], 可选) - 桶的规格：如果为int，为两个维度指定桶的数量 ``(nx=ny=bins)`` 。如果为array_like，指定两个维度的桶的边界 ``(x_edges=y_edges=bins)`` 。 如果为[int, int]，指定每个维度的桶的数量 ``(nx, ny = bins)`` 。 如果为[array, array]，指定每个维度的桶的边界 ``(x_edges, y_edges = bins)`` 。 如果为组合[int, array]或[array, int]，其中int是桶的数量，array是桶的边界。默认值： `10` 。
        - **range** (Union[list, tuple], 可选) - shape为(2, 2)的数组，指定每个维度的桶的最左和最右边界(如果在 `bins` 参数中没有明确指定)： ``[[xmin, xmax], [ymin, ymax]]`` 。 范围之外的所有值都将被视为异常值，不计入直方图。默认值： ``None`` 。
        - **weights** (Union[list, tuple, Tensor], 可选) - 一个shape为 ``(N,)`` 的数组，包含每个样本 ``(x_i, y_i)`` 的权重 `w_i` 。默认值： ``None`` 。
        - **density** (boolean, 可选) - 如果为False，默认返回每个桶中的样本计数。 如果为True，则返回桶位置处的概率密度函数值， ``bin_count / sample_count / bin_volume`` 。默认值： ``False`` 。

    返回：
        (Tensor, Tensor, Tensor)，双向直方图的值及其在第一和第二维度上的桶的边界。
