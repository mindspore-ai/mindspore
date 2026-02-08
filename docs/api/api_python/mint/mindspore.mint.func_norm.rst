mindspore.mint.norm
===================

.. py:function:: mindspore.mint.norm(input, p='fro', dim=None, keepdim=False, *, dtype=None)

    返回给定tensor的矩阵范数或向量范数。

    `p` 为范数的计算模式。支持下列范数模式。

    =================   ================================== ==============================================
    `p`                  矩阵范数                               向量范数
    =================   ================================== ==============================================
    `'fro'`              Frobenius 范数                      不支持
    `'nuc'`              Nuclear 范数                        不支持
    其余int或float值       不支持                             :math:`sum(abs(x)^{p})^{(1 / p)}`
    =================   ================================== ==============================================

    参数：
        - **input** (Tensor) - 输入tensor。
        - **p** (Union[bool, int, float, inf, -inf, 'fro', 'nuc'], 可选) - 指定要采用的范数类型。默认 ``'fro'`` 。
        - **dim** (Union[int, List(int), Tuple(int)], 可选) - 指定计算维度。默认 ``None`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果设置此参数，则会在执行之前将 `input` 转换为指定的类型。默认 ``None`` 。

    返回：
        Tensor

    .. note::
        - 动态shape、动态rank和可变输入不支持在 `图模式(mode=mindspore.GRAPH_MODE)
          <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下执行。
        - 大小值域输入情况，Ascend后端的计算结果可能出现精度误差。
