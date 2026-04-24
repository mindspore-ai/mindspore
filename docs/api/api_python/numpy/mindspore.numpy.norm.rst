mindspore.numpy.norm
====================

.. py:function:: mindspore.numpy.norm(x, ord=None, axis=None, keepdims=False)

    矩阵或向量的范数。根据 `ord` 参数的值，此函数能够返回八种不同的矩阵范数之一，或者无限多种向量范数之一(见下文)。

    .. note::
        不支持计算矩阵的核范数和二范数。

    参数：
        - **x** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。如果 `axis` 为 ``None`` ，则 `x` 必须是1-D或2-D，除非 `ord` 为 ``None`` 。如果 `axis` 和 `ord` 都为 ``None`` ，则返回 `x.ravel` 的二范数。
        - **ord** (Union[None, 'fro', 'nuc', inf, -inf, int, float], 可选) - 范数的阶数。inf表示NumPy中的inf对象。默认值: ``None`` 。
        - **axis** (Union[int, 2-tuple(int), None], 可选) - 如果 `axis` 是整数，则指定计算向量范数所沿的 `x` 的轴。如果 `axis` 是2-tuple，则指定保存2-D矩阵的轴，并计算这些矩阵的矩阵范数。如果 `axis` 为 ``None`` ，则返回向量范数(当 `x` 为1-D时)或矩阵范数(当 `x` 为2-D时)。默认为 ``None`` 。
        - **keepdims** (boolean, 可选) - 默认值:  ``False`` 。如果设置为 ``True`` ，减少的轴在结果中保留为大小为1的维度。若使用此选项，结果会广播到和 `x` 同一个维度数。

    返回：
        Tensor。 矩阵或向量的范数。

    异常：
        - **ValueError** - 如果范数的阶数未定义。