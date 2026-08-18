mindspore.mint.linalg.matrix_norm
=================================

.. py:function:: mindspore.mint.linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None)

    返回给定Tensor在指定维度上的矩阵范数。

    `ord` 为范数的计算模式。支持下列范数模式。

    ====================   ==================================
    `ord`                   矩阵范数
    ====================   ==================================
    ``'fro'`` （默认值）     Frobenius 范数
    ``'nuc'``               Nuclear 范数
    ``inf``                 :math:`\max(\sum(abs(x), dim=1))`
    ``-inf``                :math:`\min(\sum(abs(x), dim=1))`
    ``1``                   :math:`\max(\sum(abs(x), dim=0))`
    ``-1``                  :math:`\min(\sum(abs(x), dim=0))`
    ``2``                   最大奇异值
    ``-2``                  最小奇异值
    ====================   ==================================

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **A** (Tensor) - shape为 :math:`(*, m, n)` 的Tensor，其中*是零个或多个batch维度。
        - **ord** (Union[int, inf, -inf, 'fro', 'nuc'], 可选) - 范数的计算模式。行为参考上表。默认值： ``'fro'`` 。
        - **dim** (Tuple[int, int], 可选) - 计算矩阵范数的维度。默认值： ``(-2, -1)`` 。
        - **keepdim** (bool, 可选) - 输出Tensor是否保留原有的维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果设置此参数，则会在执行之前将 `A` 转换为指定的类型，返回的Tensor类型也将为指定类型 `dtype`。
          如果 `dtype` 为 ``None`` ，保持 `A` 的类型不变。默认值： ``None`` 。

    返回：
        Tensor，在指定维度 `dim` 上进行范数计算的结果。

    异常：
        - **TypeError** - `dim` 不是由int组成的tuple。
        - **ValueError** - `dim` 的长度不是2。
        - **ValueError** - `ord` 不在[2, -2, 1, -1, float('inf'), float('-inf'), 'fro', 'nuc']中。
        - **ValueError** - `dim` 的两个元素在标准化过后取值相同。
        - **ValueError** - `dim` 的任意元素超出索引。

    .. note::
        动态shape、动态rank和可变输入不支持在 `图模式(mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下执行。