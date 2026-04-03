mindspore.ops.meshgrid
======================

.. py:function:: mindspore.ops.meshgrid(*inputs, indexing='xy')

    从一维输入生成多维坐标网格。

    .. warning::
        2.9.0（不含）之后版本中， `indexing` 默认值将从 ``'xy'`` 改为 ``None`` （等同于 ``ij`` ）。

    .. note::
        - graph mode下的 `inputs` 为N个一维tensor，N应大于1。
        - pynative mode下的 `inputs` 为N个零维或一维tensor，N应大于0。数据类型为Number。
        - 对于长度为 `M` 和 `N` 的二维输入，取值为 ``'xy'`` 时，输出的shape为 :math:`(N, M)` ，取值为 ``'ij'`` 时，输出的shape为 :math:`(M, N)` 。
        - 对于长度为 `M` ， `N` 和 `P` 的三维输入，取值为 ``'xy'`` 时，输出的shape为 :math:`(N, M, P)` ，取值为 ``'ij'`` 时，输出的shape为 :math:`(M, N, P)` 。

    参数：
        - **inputs** (Union[tuple[Tensor], list[Tensor]]) - 输入tensors。

    关键字参数：
        - **indexing** (str, 可选) - 输出的网格矩阵的size。可选 ``'xy'`` 或 ``'ij'`` 。默认 ``'xy'`` 。

    返回：
        由N个N维tensor组成的元组。
