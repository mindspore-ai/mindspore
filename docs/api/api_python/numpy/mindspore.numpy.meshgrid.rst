mindspore.numpy.meshgrid
=================================

.. py:function:: mindspore.numpy.meshgrid(*xi, sparse=False, indexing='xy')

    返回由坐标向量生成的坐标矩阵。根据一维坐标数组 ``x1, x2, ... , xn`` ，生成用于对 ``N`` 维标量/向量场在 ``N`` 维网格上进行矢量化评估的 ``N`` 维坐标数组。

    .. note::
        不支持Numpy参数副本，并且始终只返回一个副本。

    参数：
        - ***xi** (Tensor) - 表示网格点坐标的一维数组。
        - **indexing** ('xy', 'ij', 可选) - 输出数组的笛卡尔（'xy'，默认值）或矩阵（'ij'）的索引。在输入长度为 ``M`` 和 ``N`` 的二维情况下，'xy'索引的输出shape为 ``(N, M)`` ，'ij'索引的输出shape为 ``(M, N)`` 。在输入长度为 ``M`` 、 ``N`` 和 ``P`` 的三维情况下，'xy'索引的输出shape为 ``(N, M, P)`` ，'ij'索引的输出shape为 ``(M, N, P)`` 。
        - **sparse** (bool, 可选) - 如果为 ``True`` 则返回稀疏矩阵以节省内存，默认值： ``False`` 。

    返回：
        元素为Tensor的Tuple，对于长度为 ``Ni=len(xi)`` 的向量 ``x1, x2, ..., xn`` ，如果 ``indexing='ij'`` ，则返回shape为 ``(N1, N2, N3,...Nn)`` 的数组，或者如果 ``indexing='xy'`` ，则返回shape为 ``(N2, N1, N3, ...Nn)`` 的数组，其中 ``xi`` 的元素沿着第一维填充 ``x1`` 矩阵，第二维度填充 ``x2`` 矩阵，依此类推。

    异常：
        - **TypeError** - 如果输入不是Tensor，或者 ``sparse`` 不是布尔值，或者 ``indexing`` 不是'xy'或'ij'。