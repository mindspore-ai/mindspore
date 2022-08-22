mindspore.nn.probability.distribution.Beta
================================================

.. py:class:: mindspore.nn.probability.distribution.Beta(concentration1=None, concentration0=None, seed=None, dtype=mstype.float32, name='Beta')

    Beta 分布（Beta Distribution）。
    连续随机分布，取值范围为 :math:`[0, 1]` ，概率密度函数为

    .. math::
        f(x, \alpha, \beta) = x^\alpha (1-x)^{\beta - 1} / B(\alpha, \beta).

    其中 :math:`B` 为 Beta 函数。

    参数：
        - **concentration1** (int, float, list, numpy.ndarray, Tensor) - Beta 分布的alpha。默认值：None。
        - **concentration0** (int, float, list, numpy.ndarray, Tensor) - Beta 分布的beta。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
        - **dtype** (mindspore.dtype) - 采样结果的数据类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'Beta'。

    .. note::
        - `concentration1` 和 `concentration0` 中元素必须大于零。
        - `dist_spec_args` 是 `concentration1` 和 `concentration0`。
        - `dtype` 必须是float，因为 Beta 分布是连续的。

    异常：
        - **ValueError** - `concentration1` 或者 `concentration0` 中元素小于0。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: concentration0
        :property:

        返回concentration0（也称为 Beta 分布的 beta）。

        返回：
            Tensor，concentration0 的值。

    .. py:method:: concentration1
        :property:

        返回concentration1（也称为 Beta 分布的 alpha）。

        返回：
            Tensor，concentration1 的值。

    .. py:method:: cdf(value, concentration1, concentration0)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, concentration1_b, concentration0_b, concentration1, concentration0)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **concentration1_b** (Tensor) - 对比 Beta 分布的 alpha。
            - **concentration0_b** (Tensor) - 对比 Beta 分布的 beta。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(concentration1, concentration0)

        计算熵。

        参数：
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，熵的值。

    .. py:method:: kl_loss(dist, concentration1_b, concentration0_b, concentration1, concentration0)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **concentration1_b** (Tensor) - 对比 Beta 分布的 alpha。
            - **concentration0_b** (Tensor) - 对比 Beta 分布的 beta。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, concentration1, concentration0)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, concentration1, concentration0)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, concentration1, concentration0)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(concentration1, concentration0)

        计算期望。

        参数：
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(concentration1, concentration0)

        计算众数。

        参数：
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, concentration1, concentration0)

        计算给定值下的概率。对于连续是计算概率密度函数（Probability Density Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, concentration1, concentration0)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(concentration1, concentration0)

        计算标准差。

        参数：        
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, concentration1, concentration0)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(concentration1, concentration0)

        计算方差。

        参数：
            - **concentration1** (Tensor) - Beta 分布的 alpha。默认值：None。
            - **concentration0** (Tensor) - Beta 分布的 beta。默认值：None。

        返回：
            Tensor，概率分布的方差。
