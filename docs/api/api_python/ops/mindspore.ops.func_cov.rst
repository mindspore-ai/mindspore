mindspore.ops.cov
==================

.. py:function:: mindspore.ops.cov(x, *, correction=1, fweights=None, aweights=None)

    给定输入 `x` 和权重，返回输入 `x` 的协方差矩阵(每对变量的协方差的方阵)，其中输入行是变量，列是观察值。

    对角线包含每个变量及其自身的协方差。如果 `x` 是单个变量的标量或一维向量，则将返回其方差。

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

    参数：
        - **x** (Tensor) - 一个二维矩阵，或单个变量的标量或一维向量。

    关键字参数：
        - **correction** (int，可选) - 样本量和样本自由度之间的差异，默认为Bessel校正 `correction = 1`，即使指定了 `fweights` 和 `aweights` 的情况下它也会返回无偏估计。`correction = 0` 将返回简单平均值。默认值：1。
        - **fweights** (Tensor, 可选) - 包含整数频率权重的标量或一维Tensor，表示每一个观测向量的重复次数。它的numel必须等于输入 `x` 的列数。若为None则忽略。默认值：None。
        - **aweights** (Tensor, 可选) - 包含浮点观测权重的标量或一维Tensor，表示每一个观测向量的重要性，重要性越高对应值越大。它的numel必须等于输入 `x` 的列数。若为None则忽略。默认值：None。

    返回：
        Tensor，输入 `x` 的协方差矩阵。

    异常：
        - **ValueError** - 如果输入的维度大于2。
        - **ValueError** - 如果 `fweights` 的维度大于1。
        - **ValueError** - 如果 `fweights` 的numel不等于输入 `x` 的列数。
        - **ValueError** - 如果 `aweights` 的numel不等于输入 `x` 的列数。
        - **ValueError** - 如果 `aweights` 的维度大于1。
        - **TypeError** - 如果输入的类型为bool类型。
        - **TypeError** - 如果 `fweights` 的类型不为int。
        - **TypeError** - 如果 `aweights` 的类型不为浮点类型。
