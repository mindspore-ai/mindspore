mindspore.ops.matrix_norm
=========================

.. py:function:: mindspore.ops.matrix_norm(A, ord='fro', axis=(-2, -1), keepdims=False, *, dtype=None)

    返回给定Tensor在指定维度上的矩阵范数。

    `ord` 为norm的计算模式。支持下列norm模式。

    =================   ==================================
    `ord`                矩阵范数
    =================   ==================================
    ``'fro'`` (默认值)    Frobenius norm
    ``'nuc'``            nuclear norm
    ``inf``              :math:`max(sum(abs(x), dim=1))`
    ``-inf``             :math:`min(sum(abs(x), dim=1))`
    ``1``                :math:`max(sum(abs(x), dim=0))`
    ``-1``               :math:`min(sum(abs(x), dim=0))`
    ``2``                最大奇异值
    ``-2``               最小奇异值
    =================   ==================================

    参数：
        - **A** (Tensor) - shape为 :math:`(*, m, n)` 的Tensor，其中*是零个或多个batch维度。
        - **ord** (Union[int, inf, -inf, 'fro', 'nuc'], 可选) - norm的模式。行为参考上表。默认值： ``None`` 。
        - **axis** (Tuple(int, int), 可选) - 计算矩阵范数的维度。默认值： ``(-2, -1)`` 。
        - **keepdims** (bool) - 输出Tensor是否保留原有的维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果设置此参数，则会在执行之前将 `A` 转换为指定的类型，返回的Tensor类型也将为指定类型 `dtype`。
          如果 `dtype` 为 ``None`` ，保持 `A` 的类型不变。默认值： ``None`` 。

    返回：
        Tensor，在指定维度 `axis` 上进行范数计算的结果，与输入 `A` 的数据类型相同。

    异常：
        - **TypeError** - `axis` 不是由int组成的tuple。
        - **ValueError** - `axis` 的长度不是2。
        - **ValueError** - `ord` 不在[2, -2, 1, -1, float('inf'), float('-inf'), 'fro', 'nuc']中。
        - **ValueError** - `axis` 的两个元素在标准化过后取值相同。
        - **ValueError** - `axis` 的任意元素超出索引。
