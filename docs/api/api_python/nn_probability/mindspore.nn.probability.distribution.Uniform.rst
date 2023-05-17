mindspore.nn.probability.distribution.Uniform
================================================

.. py:class:: mindspore.nn.probability.distribution.Uniform(low=None, high=None, seed=None, dtype=mstype.float32, name='Uniform')

    均匀分布（Uniform Distribution）。
    连续随机分布，取值范围为 :math:`[a, b]` ，概率密度函数为

    .. math:: 
        f(x, a, b) = 1 / (b - a).

    其中 :math:`a, b` 为分别为均匀分布的下界和上界。

    参数：
        - **low** (int, float, list, numpy.ndarray, Tensor) - 分布的下限。默认值： ``None`` 。
        - **high** (int, float, list, numpy.ndarray, Tensor) - 分布的上限。默认值： ``None`` 。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值： ``None`` 。
        - **dtype** (mindspore.dtype) - 事件样例的类型。默认值： ``mstype.float32`` 。
        - **name** (str) - 分布的名称。默认值： ``'Uniform'`` 。

    .. note:: 
        - `low` 必须小于 `high` 。
        - `dtype` 必须是float类型，因为均匀分布是连续的。


    异常：
        - **ValueError** - `low` 大于等于 `high` 。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: high
        :property:

        返回分布的上限。

        返回：
            Tensor，分布的上限值。

    .. py:method:: low
        :property:

        返回分布的下限。

        返回：
            Tensor，分布的下限值。

    .. py:method:: cdf(value, high, low)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, high_b, low_b, high, low)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **high_b** (Tensor) - 对比分布的上限值。
            - **low_b** (Tensor) - 对比分布的下限值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(high, low)

        计算熵。

        参数：
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，熵的值。

    .. py:method:: kl_loss(dist, high_b, low_b, high, low)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **high_b** (Tensor) - 对比分布的上限值。
            - **low_b** (Tensor) - 对比分布的下限值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, high, low)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, high, low)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, high, low)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(high, low)

        计算期望。

        参数：
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(high, low)

        计算众数。

        参数：
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, high, low)

        计算给定值下的概率。对于连续是计算概率密度函数（Probability Density Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, high, low)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(high, low)

        计算标准差。

        参数：        
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, high, low)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(high, low)

        计算方差。

        参数：
            - **high** (Tensor) - 分布的上限值。默认值： ``None`` 。
            - **low** (Tensor) - 分布的下限值。默认值： ``None`` 。

        返回：
            Tensor，概率分布的方差。
