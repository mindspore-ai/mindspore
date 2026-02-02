mindspore.numpy.tril
=================================

.. py:function:: mindspore.numpy.tril(m, k=0)

    返回张量的下三角部分。
    返回Tensor的副本，其中第 `k` 个对角线以上的元素为0。

    参数：
        - **m** (Union[Tensor, list, tuple]) - 输入的Tensor。返回的Tensor与 `m` 的shape、数据类型都相同。
        - **k** (int, 可选) - 对角线的偏移量： :math:`k=0` 即为主对角线， :math:`k<0` 即对角线向下偏移， :math:`k>0` 即对角线向上偏移。默认值： ``0`` 。

    返回：
        Tensor， `m` 的下三角部分，shape和数据类型与 `m` 相同。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果输入的 `m` 的秩 :math:`<1` 。