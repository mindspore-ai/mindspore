mindspore.nn.probability.distribution.Exponential
===================================================

.. py:class:: mindspore.nn.probability.distribution.Exponential(rate=None, seed=None, dtype=mstype.float32, name='Exponential')

    指数分布（Exponential Distribution）。
    连续随机分布，取值范围为所有正实数 :math:`[0, \inf)`，概率密度函数为

    .. math::
        f(x, \lambda) = \lambda \exp(-\lambda x).

    其中 :math:`\lambda` 为分别为指数分布的率参数。

    参数：
        - **rate** (int, float, list, numpy.ndarray, Tensor) - 率参数。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
        - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'Exponential'。

    .. note:: 
        - `rate` 中的元素必须大于0。
        - `dtype` 必须是float，因为指数分布是连续的。

    异常：
        - **ValueError** - `rate` 中元素小于0。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: rate
        :property:

        返回 `rate` 。

        返回：
            Tensor，率参数的值。

    .. py:method:: cdf(value, rate)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, rate_b, rate)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **rate_b** (Tensor) - 对比分布的率参数。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(rate)

        计算熵。

        参数：
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，熵的值。

    .. py:method:: kl_loss(dist, rate_b, rate)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **rate_b** (Tensor) - 对比分布的率参数。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, rate)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, rate)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, rate)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(rate)

        计算期望。

        参数：
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(rate)

        计算众数。

        参数：
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, rate)

        计算给定值下的概率。对于连续是计算概率密度函数（Probability Density Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, rate)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(rate)

        计算标准差。

        参数：        
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, rate)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(rate)

        计算方差。

        参数：
            - **rate** (Tensor) - 分布的率参数。默认值：None。

        返回：
            Tensor，概率分布的方差。
