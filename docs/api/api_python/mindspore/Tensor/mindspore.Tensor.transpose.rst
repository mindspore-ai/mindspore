mindspore.Tensor.transpose
==========================

.. py:method:: mindspore.Tensor.transpose(dim0, dim1) -> Tensor

    通过给定的维度对输入Tensor进行转置。

    参数：
        - **dim0** (int) - 指定第一个需要被转置的维度。
        - **dim1** (int) - 指定第二个需要被转置的维度。

    返回：
        转置后的Tensor，与输入具有相同的数据类型。

    异常：
        - **TypeError** - 如果 `dim0` 或者 `dim1` 的数据类型不是int。
        - **ValueError** - 如果 `dim0` 或者 `dim1` 的数值超出了范围：:math:`[-ndim, ndim-1]` 。

    .. py:method:: mindspore.Tensor.transpose(*axes) -> Tensor
        :noindex:

    根据指定的排列对输入的Tensor进行数据重排。

    此函数对于一维数组转置后不产生变化。对于一维数组转为二维列向量，请参照：:func:`mindspore.ops.expand_dims` 。对于二维数组可以看做是标准的矩阵转置。对于n维数组，根据指定的轴进行排列。如果没有指定轴并且a.shape为： :math:`(i[0], i[1], ... i[n-2], i[n-1])` ，那么a.transpose().shape为： :math:`(i[n-1], i[n-2], ... i[1], i[0])` 。

    .. note::
        GPU和CPU平台上，如果 `axes` 的元素值为负数，则其实际值为 `axes[i] + rank(self)` 。

    参数：
        - **axes** (tuple[int]) - 指定转置的排列。 `axes` 中的元素由 `self` 的每个维度的索引组成。 `axes` 的长度和 `self` 的shape相同。只支持常量值。其范围在 `[-rank(self), rank(self))` 内。

    返回：
        Tensor。输出Tensor的数据类型与输入相同，输出Tensor的shape由 `self` 的shape和 `axes` 的值决定。

    异常：
        - **TypeError** - 如果 `axes` 的类型不为tuple。
        - **ValueError** - 如果 `self` 的shape长度不等于 `axes` 的长度。
        - **ValueError** - 如果 `axes` 中存在相同的元素。
