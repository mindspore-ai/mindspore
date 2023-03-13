mindspore.ops.Meshgrid
======================

.. py:class:: mindspore.ops.Meshgrid(indexing='xy')

    从给定的Tensor生成网格矩阵。给定N个一维Tensor，对每个Tensor做扩张操作，返回N个N维的Tensor。

    更多参考详见 :func:`mindspore.ops.meshgrid`。

    参数：
        - **indexing** (str, 可选) - 以笛卡尔坐标'xy'或者矩阵'ij'索引作为输出。可选值为'xy'或'ij'。对于长度为 `M` 和 `N` 的二维输入，取值为'xy'时，输出的shape为 :math:`(N, M)` ，取值为'ij'时，输出的shape为 :math:`(M, N)` 。以长度为 `M` , `N` 和 `P` 的三维输入，取值为'xy'时，输出的shape为 :math:`(N, M, P)` ，取值为'ij'时，输出的shape为 :math:`(M, N, P)` 。默认值：'xy'。

    输入：
        - **input** (Union[tuple]) - N个一维Tensor。输入的长度应大于1。数据类型为Number。

    输出：
        Tensor，N个N维tensor对象的元组。数据类型与输入相同。
