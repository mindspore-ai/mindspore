mindspore.nn.probability.distribution.Gumbel
================================================

.. py:class:: mindspore.nn.probability.distribution.Gumbel(loc, scale, seed=0, dtype=mstype.float32, name='Gumbel')

    Gumbel分布（Gumbel distribution）。
    连续随机分布，取值范围为所有实数 ，概率密度函数为

    .. math:: 
        f(x, a, b) = 1 / b \exp(\exp(-(x - a) / b) - x).

    其中 :math:`a, b` 为分别为Gumbel分布的位置参数和比例参数。

    参数：
        - **loc** (int, float, list, numpy.ndarray, Tensor) - Gumbel分布的位置。
        - **scale** (int, float, list, numpy.ndarray, Tensor) - Gumbel分布的尺度。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：0。
        - **dtype** (mindspore.dtype) - 分布类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'Gumbel'。

    .. note:: 
        - `scale` 必须大于零。
        - `dtype` 必须是浮点类型，因为Gumbel分布是连续的。
        - GPU后端不支持 `kl_loss` 和 `cross_entropy` 。

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
