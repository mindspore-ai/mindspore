mindspore.numpy.histogram
=========================

.. py:function:: mindspore.numpy.histogram(a, bins=10, range=None, weights=None, density=False)

    计算数据集的直方图。

    .. note::
        不支持 `bins` 为string类型。 不支持已弃用的NumPy参数 `normed` 。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入数据。直方图在展平的数组上计算。
        - **bins** (Union[int, tuple, list, Tensor], 可选) - 如果 `bins` 为int，它定义了给定范围(默认为10)中等宽的桶的数量。 如果 `bins` 是一个序列，它定义了桶的边界，包括最右边界，允许非均匀的桶宽度。默认值： `10` 。
        - **range** ((float, float), 可选) - 桶的上下界。 如果未提供， `range` 简单定义成 ``(a.min(), a.max())`` 。 范围之外的值将被忽略。 `range` 的第一个元素必须小于等于第二个元素。默认值： ``None`` 。
        - **weights** (Union[int, float, bool, list, tuple, Tensor], 可选) - 一个与 `a` shape相同的权重数组。 如果 `density` 为True，权重将被归一化，使得范围内的密度总和为1。默认值： ``None`` 。
        - **density** (boolean, 可选) - 如果为False，结果将包含每个桶中的样本数量。 如果为True，结果是桶的位置处概率密度函数的值，规一化使得其在整个范围上的积分为1。 请注意，直方图值的总和不会等于1，除非选择了单位宽度的桶。 它不是一个概率质量函数。默认值： ``False`` 。

    返回：
        (Tensor, Tensor)，直方图的值和桶的边界。

    异常：
        - **ValueError** - 如果 `a` 和 `weights` 的size不相同。