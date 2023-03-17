mindspore.nn.probability.distribution.Gamma
================================================

.. py:class:: mindspore.nn.probability.distribution.Gamma(concentration=None, rate=None, seed=None, dtype=mstype.float32, name='Gamma')

    伽马分布（Gamma distribution）。
    连续随机分布，取值范围为 :math:`(0, \inf)` ，概率密度函数为

    .. math::
        f(x, \alpha, \beta) = \beta^\alpha / \Gamma(\alpha) x^{\alpha - 1} \exp(-\beta x).

    其中 :math:`G` 为 Gamma 函数，:math:`\alpha` 和 :math:`\beta` 为分别 Gamma 函数的浓度参数和逆尺度参数。

    参数：
        - **concentration** (int, float, list, numpy.ndarray, Tensor) - 浓度参数，也被称为伽马分布的alpha。默认值：None。
        - **rate** (int, float, list, numpy.ndarray, Tensor) - 逆尺度参数，也被称为伽马分布的beta。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
        - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'Gamma'。

    .. note:: 
        - `concentration` 和 `rate` 中的元素必须大于零。
        - `dtype` 必须是float，因为伽马分布是连续的。

    异常：
        - **ValueError** - `concentration` 或者 `rate` 中元素小于0。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: concentration
        :property:

        返回分布的浓度（也称为伽马分布的alpha）。

        返回：
            Tensor，concentration 的值。

    .. py:method:: rate
        :property:

        返回分布的逆尺度（也称为伽马分布的beta）。

        返回：
            Tensor，rate 的值。

    .. py:method:: cdf(value, concentration, rate)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, concentration_b, rate_b, concentration, rate)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **concentration_b** (Tensor) - 对比分布的alpha。
            - **rate_b** (Tensor) - 对比分布的beta。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(concentration, rate)

        计算熵。

        参数：
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，熵的值。

    .. py:method:: kl_loss(dist, concentration_b, rate_b, concentration, rate)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **concentration_b** (Tensor) - 对比分布的alpha。
            - **rate_b** (Tensor) - 对比分布的beta。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, concentration, rate)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, concentration, rate)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, concentration, rate)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(concentration, rate)

        计算期望。

        参数：
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(concentration, rate)

        计算众数。

        参数：
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, concentration, rate)

        计算给定值下的概率。对于连续是计算概率密度函数（Probability Density Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, concentration, rate)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(concentration, rate)

        计算标准差。

        参数：        
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, concentration, rate)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(concentration, rate)

        计算方差。

        参数：
            - **concentration** (Tensor) - 分布的alpha。默认值：None。
            - **rate** (Tensor) - 分布的beta。默认值：None。

        返回：
            Tensor，概率分布的方差。
