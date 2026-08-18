mindspore.numpy.correlate
=========================

.. py:function:: mindspore.numpy.correlate(a, v, mode='valid')

    两个一维序列的互相关。

    函数根据信号处理教材中通常定义的方式计算相关性： :math:`c_{av}[k] = \sum_{n}{a[n+k] * conj(v[n])}` 。

    其中，序列 `a` 和 `v` 在必要时将进行零填充， `conj` 为共轭。

    .. note::
        - `correlate` 目前仅用于 `mindscience` 科学计算场景，并不支持其他使用场景。
        - `correlate` 还未在Windows平台上得到支持。

    参数：
        - **a** (Union[list, tuple, Tensor]) - 第一个输入序列。
        - **v** (Union[list, tuple, Tensor]) - 第二个输入序列。
        - **mode** (str, 可选) - 指定的填充模式。可选值： ``"same"`` 、 ``"valid"`` 、 ``"full"`` 。 默认值： ``"valid"`` 。

          - ``"same"`` ：返回的输出长度为 :math:`\max(M, N)` 。仍然可见边界效应。

          - ``"valid"`` ：返回的输出长度为 :math:`\max(M, N) - \min(M, N) + 1` 。卷积乘积仅在信号完全重叠的点给出。 信号边界外的值不产生效果。

          - ``"full"`` ：在每个重叠点返回卷积，输出的shape为 :math:`(N + M - 1)`。在卷积的端点，信号不完全重叠，可能会看到边界效应。

    返回：
        Tensor， `a` 和 `v` 的离散互相关。

    异常：
        - **TypeError** - 如果 `a` 或 `v` 不是Tensor。
        - **TypeError** - 如果 `a` 和 `v` 的类型不同。
        - **ValueError** - 如果 `a` 和 `v` 为空或维数错误。