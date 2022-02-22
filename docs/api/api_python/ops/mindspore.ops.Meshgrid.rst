mindspore.ops.Meshgrid
========================

.. py:class:: mindspore.ops.Meshgrid(indexing="xy")

    从给定的Tensor生成网格矩阵。

    给定N个一维Tensor，对每个Tensor做扩张操作，返回N个N维的Tensor。

    **参数：**

    - **indexing** ('xy', 'ij', optional) - 'xy'或'ij'。默认值：'xy'。以笛卡尔坐标'xy'或者矩阵'ij'索引作为输出。以长度为 `M` 和 `N` 的二维输入，取值为'xy'时，输出的shape为 `(N, M)` ，取值为'ij'时，输出的shape为 `(M, N)` 。以长度为 `M` , `N` 和 `P` 的三维输入，取值为'xy'时，输出的shape为 `(N, M, P)` ，取值为'ij'时，输出的shape为 `(M, N, P)` 。

    **输入：**

    - **input** (Union[tuple]) - N个一维Tensor。输入的长度应大于1。数据类型为Number。

    **输出：**

    Tensor，N个N维tensor对象的元组。数据类型与输入相同。

    **异常：**

    - **TypeError** - `indexing` 不是str或 `input` 不是元组。
    - **ValueError** - `indexing` 的取值既不是'xy'也不是'ij'。