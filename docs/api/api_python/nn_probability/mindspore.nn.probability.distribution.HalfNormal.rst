mindspore.nn.probability.distribution.HalfNormal
================================================

.. py:class:: mindspore.nn.probability.distribution.HalfNormal(mean=None, sd=None, seed=None, dtype=mstype.float32, name='HalfNormal')

    半正态分布（HalfNormal distribution）。
    连续随机分布，取值范围为 :math:`[\mu, \inf)` ，概率密度函数为

    .. math:: 
        f(x; \mu, \sigma) = 1 / \sigma\sqrt{2\pi} \exp(-(x - \mu)^2 / 2\sigma^2).

    其中 :math:`\mu, \sigma` 为分别为半正态分布的期望与标准差。

    参数：
        - **mean** (int, float, list, numpy.ndarray, Tensor) - 半正态分布的平均值。默认值：None。
        - **sd** (int, float, list, numpy.ndarray, Tensor) - 半正态分布的标准差。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
        - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'HalfNormal'。

    .. note:: 
        - `sd` 必须大于0。
        - `dtype` 必须是float，因为半正态分布是连续的。

    异常：
        - **ValueError** - `sd` 中元素不大于0。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: log_prob(value, mean=None, sd=None)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **mean** (Tensor) - 分布的期望。默认值：None。
            - **sd** (Tensor) - 分布的标准差。默认值：None。

        返回：
            Tensor，概率密度函数的对数。