mindspore.numpy.matrix_power
============================

.. py:function:: mindspore.numpy.matrix_power(a, n)

    计算方阵以整数 `n` 为指数的幂。

    对于正整数 `n` ，通过重复的矩阵平方和矩阵乘法来计算幂。如果 :math:`n == 0` ，返回与 `a` 相同shape的单位矩阵。

    .. note::
        目前不支持堆叠的对象矩阵，也不支持 :math:`n < 0` 。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入矩阵。
        - **n** (int) - 指数可以是任意整数或长整数，正数或零。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入不能转换为Tensor或指数不是整数。
        - **ValueError** - 如果输入的维度少于2或最后两个维度不是方阵。
    