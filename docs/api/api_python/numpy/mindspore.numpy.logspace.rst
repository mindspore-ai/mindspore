mindspore.numpy.logspace
=================================

.. py:function:: mindspore.numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)

    返回在对数刻度上均匀间隔的值。
    在线性空间中，序列从 ``base ** start`` 开始，以 ``base ** stop`` 结束。

    参数：
        - **start** (Union[int, list(int), tuple(int), tensor]) -  ``base ** start`` 是序列的起始值。
        - **stop** (Union[int, list(int), tuple(int), tensor]) -  当 ``endpoint`` 为 ``True`` 时， ``base ** stop`` 是序列的最终值；当 ``endpoint`` 为 ``False`` 时，在对数空间内的区间上均匀间隔 num + 1 个值，返回除最后一个值外（长度为 num 的序列）的其他值。
        - **num** (int, 可选) - 要生成的等间隔样例数量，默认值： ``50`` 。
        - **endpoint** (bool, 可选) - 序列中是否包含 ``stop`` 值，默认值： ``True`` 。
        - **base** (Union[int, float], 可选) - 对数的底数。 :math:`ln(samples) / ln(base)` (或 :math:`log_{base}(samples)`) 元素之间的步长是均匀的，默认值： ``10.0`` 。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor ``dtype`` 。如果 ``dtype`` 为  ``None`` ，则将从其他输入参数推断出新Tensor的数据类型。默认值： ``None`` 。
        - **axis** (int, 可选) - 结果中用于存储样本的轴。仅当 ``start`` 或 ``stop`` 为类似数组对象时才用到。默认值： ``0`` ，默认情况下的采样将沿着在开始处插入的新轴。使用 ``-1`` 在末尾获取一个轴。

    返回：
        Tensor，对数刻度上均匀间隔的样本。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。