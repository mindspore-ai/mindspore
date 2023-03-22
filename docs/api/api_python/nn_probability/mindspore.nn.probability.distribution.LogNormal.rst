mindspore.nn.probability.distribution.LogNormal
================================================

.. py:class:: mindspore.nn.probability.distribution.LogNormal(loc=None, scale=None, seed=0, dtype=mstype.float32, name='LogNormal')

    对数正态分布（LogNormal distribution）。
    连续随机分布，取值范围为 :math:`(0, \inf)` ，概率密度函数为

    .. math:: 
        f(x, \mu, \sigma) = 1 / x\sigma\sqrt{2\pi} \exp(-(\ln(x) - \mu)^2 / 2\sigma^2).

    其中 :math:`\mu, \sigma` 为分别为基础正态分布的平均值和标准差。
    服从对数正态分布的随机变量的对数服从正态分布。它被构造为正态分布的指数变换。

    参数：
        - **loc** (int, float, list, numpy.ndarray, Tensor) - 基础正态分布的平均值。默认值：None。
        - **scale** (int, float, list, numpy.ndarray, Tensor) - 基础正态分布的标准差。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：0。
        - **dtype** (mindspore.dtype) - 分布类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'LogNormal'。

    .. note:: 
        - `scale` 必须大于零。
        - `dtype` 必须是float，因为对数正态分布是连续的。

    异常：
        - **ValueError** - `scale` 中元素小于0。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: loc
        :property:

        返回分布位置。

        返回：
            Tensor，分布的位置值。

    .. py:method:: scale
        :property:

        返回分布比例。

        返回：
            Tensor，分布的比例值。

    .. py:method:: cdf(value, loc, scale)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, loc_b, scale_b, loc, scale)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **loc_b** (Tensor) - 对比分布位置参数。
            - **scale_b** (Tensor) - 对比分布比例参数。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(loc, scale)

        计算熵。

        参数：
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，熵的值。

    .. py:method:: kl_loss(dist, loc_b, scale_b, loc, scale)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **loc_b** (Tensor) - 对比分布位置参数。
            - **scale_b** (Tensor) - 对比分布比例参数。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, loc, scale)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, loc, scale)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, loc, scale)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(loc, scale)

        计算期望。

        参数：
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(loc, scale)

        计算众数。

        参数：
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, loc, scale)

        计算给定值下的概率。对于连续是计算概率密度函数（Probability Density Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, loc, scale)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(loc, scale)

        计算标准差。

        参数：        
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, loc, scale)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(loc, scale)

        计算方差。

        参数：
            - **loc** (Tensor) - 分布位置参数。默认值：None。
            - **scale** (Tensor) - 分布比例参数。默认值：None。

        返回：
            Tensor，概率分布的方差。
