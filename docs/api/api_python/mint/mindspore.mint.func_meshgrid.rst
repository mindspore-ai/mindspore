mindspore.mint.meshgrid
=======================

.. py:function:: mindspore.mint.meshgrid(*tensors, indexing=None)

    从给定的tensors生成网格矩阵。

    输入N个一维tensor，对每个tensor做扩张操作，返回N个N维的tensor。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - Graph mode，dynamic shape场景下：仅支持输入N个一维tensor，N > 1。

    参数：
        - **tensors** (Union(tuple[Tensor], list[Tensor])) - N个零维或一维tensor，N > 0, tensor的数据类型为Number。

    关键字参数：
        - **indexing** (str, 可选) - 影响输出的网格矩阵的size。可选值为： ``'xy'`` 或 ``'ij'`` 。对于长度为 `M` 和 `N` 的二维输入，取值为 ``'xy'`` 时，输出的shape为 :math:`(N, M)` ，取值为 ``'ij'`` 时，输出的shape为 :math:`(M, N)` 。以长度为 `M` ， `N` 和 `P` 的三维输入，取值为 ``'xy'`` 时，输出的shape为 :math:`(N, M, P)` ，取值为 ``'ij'`` 时，输出的shape为 :math:`(M, N, P)` 。默认 ``None`` ，此时等价于取值为 ``'ij'`` 。

    返回：
        Tensors，N个N维tensor对象的元组。数据类型与输入相同。

    异常：
        - **ValueError** - `indexing` 的取值既不是 ``'xy'`` 也不是 ``'ij'`` 。
