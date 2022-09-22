mindspore.nn.probability.distribution.Poisson
================================================

.. py:class:: mindspore.nn.probability.distribution.Poisson(rate=None, seed=None, dtype=mstype.float32, name='Poisson')

    泊松分布（Poisson Distribution）。
    离散随机分布，取值范围为正自然数集，概率质量函数为

    .. math::
        P(X = k) = \lambda^k \exp(-\lambda) / k!, k = 1, 2, ...

    其中 :math:`\lambda` 为率参数(rate)。

    参数：
        - **rate** (list, numpy.ndarray, Tensor) - 泊松分布的率参数。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
        - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'Poisson'。

    .. note:: 
        `rate` 必须大于0。 `dist_spec_args` 是 `rate`。

    异常：
        - **ValueError** - `rate` 中元素小于0。

    .. py:method:: rate
        :property:

        返回分布的 `rate` 参数。

        返回：
            Tensor，rate 参数的值。

    .. py:method:: cdf(value, rate)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: log_cdf(value, rate)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, rate)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, rate)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(rate)

        计算期望。

        参数：
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(rate)

        计算众数。

        参数：
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, rate)

        计算给定值下的概率。对于离散分布是计算概率质量函数（Probability Mass Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, rate)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(rate)

        计算标准差。

        参数：        
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, rate)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(rate)

        计算方差。

        参数：
            - **rate** (Tensor) - 率参数(rate)。默认值：None。

        返回：
            Tensor，概率分布的方差。
