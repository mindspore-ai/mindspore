mindspore.mint.meshgrid
=======================

.. py:function:: mindspore.mint.meshgrid(*tensors, indexing=None)

    从给定的Tensor生成网格矩阵。

    给定N个一维Tensor，对每个Tensor做扩张操作，返回N个N维的Tensor。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **tensors** (Union(tuple[Tensor], list[Tensor])) - 静态图下为N个一维Tensor，输入的Tensor个数应大于1。动态图下为N个零维或一维Tensor，输入的Tensor个数应大于0。数据类型为Number。

    关键字参数：
        - **indexing** (str, 可选) - 影响输出的网格矩阵的size。可选值为： ``'xy'`` 或 ``'ij'`` 。对于长度为 `M` 和 `N` 的二维输入，取值为 ``'xy'`` 时，输出的shape为 :math:`(N, M)` ，取值为 ``'ij'`` 时，输出的shape为 :math:`(M, N)` 。以长度为 `M` ， `N` 和 `P` 的三维输入，取值为 ``'xy'`` 时，输出的shape为 :math:`(N, M, P)` ，取值为 ``'ij'`` 时，输出的shape为 :math:`(M, N, P)` 。默认值：``None`` ，此时等价于取值为 ``'ij'`` 。

    返回：
        Tensor，N个N维tensor对象的元组。数据类型与输入相同。

    异常：
        - **TypeError** - `indexing` 不是str或 `tensors` 不是元组。
        - **ValueError** - `indexing` 的取值既不是 ``'xy'`` 也不是 ``'ij'`` 。
