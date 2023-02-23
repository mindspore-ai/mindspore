mindspore.ops.transpose
=======================

.. py:function:: mindspore.ops.transpose(input_x, input_perm)

    根据指定的排列对输入的Tensor进行数据重排。

    此函数对于一维数组转置后不产生变化。对于一维数组转为二维列向量，请参照： :class:`mindspore.ops.ExpandDims` 。对于二维数组可以看做是标准的矩阵转置。对于n维数组，根据指定的轴进行排列。如果没有指定轴并且a.shape为 :math:`(i[0], i[1], ... i[n-2], i[n-1])` ，那么a.transpose().shape为 :math:`(i[n-1], i[n-2], ... i[1], i[0])` 。

    .. note::
        GPU和CPU平台上，如果 `input_perm` 的元素值为负数，则其实际值为 `input_perm[i] + rank(input_x)` 。 Ascend平台不支持 `input_perm` 元素值为负。

    参数：
        - **input_x** (Tensor) - 输入Tensor，其shape是 :math:`(x_1, x_2, ..., x_R)` 。
        - **input_perm** (tuple[int]) - 指定排列。 `input_perm` 中的元素由 `input_x` 的每个维度的索引组成。 `input_perm` 的长度和 `input_x` 的shape相同。只支持常量值。其范围在[-rank(input_x), rank(input_x))内。

    返回：
        Tensor，输出Tensor的数据类型与 `input_x` 相同，输出Tensor的shape由 `input_x` 的shape和 `input_perm` 的值决定。

    异常：
        - **TypeError** - `input_perm` 不是tuple。
        - **ValueError** - `input_x` 的shape长度不等于 `input_perm` 的shape长度。
        - **ValueError** - `input_perm` 中存在相同的元素。