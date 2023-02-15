mindspore.ops.meshgrid
======================

.. py:function:: mindspore.ops.meshgrid(*inputs, indexing='xy')

    从给定的Tensor生成网格矩阵。

    给定N个一维Tensor，对每个Tensor做扩张操作，返回N个N维的Tensor。

    参数：
        - **inputs** (List[Tensor]) - N个一维Tensor的List。输入的长度应大于1。数据类型为Number。

    关键字参数：
        - **indexing** ('xy', 'ij', 可选) - 'xy'或'ij'。影响输出的网格矩阵的size。对于长度为 `M` 和 `N` 的二维输入，取值为'xy'时，输出的shape为 :math:`(N, M)` ，取值为'ij'时，输出的shape为 :math:`(M, N)` 。以长度为 `M` ， `N` 和 `P` 的三维输入，取值为'xy'时，输出的shape为 :math:`(N, M, P)` ，取值为'ij'时，输出的shape为 :math:`(M, N, P)` 。默认值：'xy'。

    返回：
        Tensor，N个N维tensor对象的元组。数据类型与输入相同。

    异常：
        - **TypeError** - `indexing` 不是str或 `inputs` 不是元组。
        - **ValueError** - `indexing` 的取值既不是'xy'也不是'ij'。
