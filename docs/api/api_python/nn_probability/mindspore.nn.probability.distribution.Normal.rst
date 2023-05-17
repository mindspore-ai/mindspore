mindspore.nn.probability.distribution.Normal
================================================

.. py:class:: mindspore.nn.probability.distribution.Normal(mean=None, sd=None, seed=None, dtype=mstype.float32, name='Normal')

    正态分布（Normal distribution）。
    连续随机分布，取值范围为 :math:`(-\inf, \inf)` ，概率密度函数为

    .. math:: 
        f(x, \mu, \sigma) = 1 / \sigma\sqrt{2\pi} \exp(-(x - \mu)^2 / 2\sigma^2).

    其中 :math:`\mu, \sigma` 为分别为正态分布的期望与标准差。

    参数：
        - **mean** (int, float, list, numpy.ndarray, Tensor) - 正态分布的平均值。默认值： ``None`` 。
        - **sd** (int, float, list, numpy.ndarray, Tensor) - 正态分布的标准差。默认值： ``None`` 。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值： ``None`` 。
        - **dtype** (mindspore.dtype) - 事件样例的类型。默认值： ``mstype.float32`` 。
        - **name** (str) - 分布的名称。默认值： ``'Normal'`` 。

    .. note:: 
        - `sd` 必须大于零。
        - `dtype` 必须是float，因为正态分布是连续的。

    异常：
        - **ValueError** - `sd` 中元素小于0。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: mean
        :property:

        返回分布期望。

        返回：
            Tensor，分布的期望。

    .. py:method:: sd
        :property:

        返回分布的标准差。

        返回：
            Tensor，分布的标准差。

    .. py:method:: cdf(value, mean, sd)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, mean_b, sd_b, mean, sd)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **mean_b** (Tensor) - 对比分布的期望。
            - **sd_b** (Tensor) - 对比分布的标准差。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(mean, sd)

        计算熵。

        参数：
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，熵的值。

    .. py:method:: kl_loss(dist, mean_b, sd_b, mean, sd)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **mean_b** (Tensor) - 对比分布的期望。
            - **sd_b** (Tensor) - 对比分布的标准差。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, mean, sd)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, mean, sd)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, mean, sd)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mode(mean, sd)

        计算众数。

        参数：
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, mean, sd)

        计算给定值下的概率。对于连续是计算概率密度函数（Probability Density Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, mean, sd)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: survival_function(value, mean, sd)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(mean, sd)

        计算方差。

        参数：
            - **mean** (Tensor) - 分布的期望。默认值： ``None`` 。
            - **sd** (Tensor) - 分布的标准差。默认值： ``None`` 。

        返回：
            Tensor，概率分布的方差。
