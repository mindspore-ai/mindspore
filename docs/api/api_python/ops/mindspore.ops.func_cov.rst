mindspore.ops.cov
==================

.. py:function:: mindspore.ops.cov(input, *, correction=1, fweights=None, aweights=None)

    返回输入tensor的协方差矩阵，其中输入tensor的行表示变量，列表示观测值。协方差矩阵的对角线为输入tensor每个变量的方差，非对角线上的元素为两两变量之间的协方差。
    当输入为零维或一维tensor时，返回其方差。

    变量 :math:`a` 和 :math:`b` 的无偏样本协方差由下式给出：

    .. math::
        \text{cov}_w(a,b) = \frac{\sum^{N}_{i = 1}(a_{i} - \bar{a})(b_{i} - \bar{b})}{N~-~1}

    其中 :math:`\bar{a}` 和 :math:`\bar{b}` 分别是 :math:`a` 和 :math:`b` 的简单均值。

    如果提供了 `fweights` 和/或 `aweights` ，则计算无偏加权协方差，由下式给出：

    .. math::
        \text{cov}_w(a,b) = \frac{\sum^{N}_{i = 1}w_i(a_{i} - \mu_a^*)(b_{i} - \mu_b^*)}{\sum^{N}_{i = 1}w_i~-~1}

    其中 :math:`w` 基于提供的 `fweights` 或 `aweights` 中的任意一个参数进行表示，如果两个参数都有提供，则 :math:`w = fweights \times aweights`，并且 :math:`\mu_x^* = \frac{\sum^{N}_{i = 1}w_ix_{i} }{\sum^{N}_{i = 1}w_i}` 表示变量的加权平均值。

    .. warning::
        `fweights` 和 `aweights` 的值不能为负数，负数权重场景结果未定义。

    .. note::
        当前暂不支持复数。

    参数：
        - **input** (Tensor) - 零维、一维或二维输入tensor。

    关键字参数：
        - **correction** (int，可选) - 样本大小与样本自由度之间的差值。`correction = 0` 将返回简单平均值。默认 ``1`` ，表示Bessel校正，即使指定了 `fweights` 和 `aweights` ，也会返回无偏估计。
        - **fweights** (Tensor, 可选) - 标量或一维tensor，表示每一个观测向量的重复次数（频率）。必须为整数类型。元素数必须等于输入 `input` 的列数。若为None则忽略。默认 ``None`` 。
        - **aweights** (Tensor, 可选) - 标量或一维tensor，表示每一个观测向量的重要性（权重），重要性越高对应值越大。必须为浮点数类型。元素数必须等于输入 `input` 的列数。若为None则忽略。默认 ``None`` 。

    返回：
        Tensor
