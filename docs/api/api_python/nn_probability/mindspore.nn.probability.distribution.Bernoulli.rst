mindspore.nn.probability.distribution.Bernoulli
================================================

.. py:class:: mindspore.nn.probability.distribution.Bernoulli(probs=None, seed=None, dtype=mstype.int32, name='Bernoulli')

    伯努利分布（Bernoulli Distribution）。
    离散随机分布，取值范围为 :math:`\{0, 1\}` ，概率质量函数为 :math:`P(X = 0) = p, P(X = 1) = 1-p`。

    参数：
        - **probs** (float, list, numpy.ndarray, Tensor) - 结果是1的概率。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
        - **dtype** (mindspore.dtype) - 采样结果的数据类型。默认值：mstype.int32.
        - **name** (str) - 分布的名称。默认值：'Bernoulli'。

    .. note:: 
        `probs` 中元素必须是合适的概率（0<p<1）。`dist_spec_args` 是 `probs`。

    异常：
        - **ValueError** - `probs` 中元素小于0或大于1。

    .. py:method:: probs
        :property:

        返回结果为1的概率。

        返回：
            Tensor，结果为1的概率。

    .. py:method:: cdf(value, probs1)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: cross_entropy(dist, probs1_b, probs1_a)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **probs1_b** (Tensor) - 对比分布的伯努利实验成功的概率。
            - **probs1_a** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(probs1)

        计算熵。

        参数：
            - **probs1** (Tensor) - 对比分布的伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，熵的值。

    .. py:method:: kl_loss(dist, probs1_b, probs1_a)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **probs1_b** (Tensor) - 对比分布的伯努利实验成功的概率。
            - **probs1_a** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, probs1)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, probs1)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, probs1)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(probs1)

        计算期望。

        参数：
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(probs1)

        计算众数。

        参数：
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, probs1)

        计算给定值下的概率。对于离散分布是计算概率质量函数（Probability Mass Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape, probs1)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(probs1)

        计算标准差。

        参数：        
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, probs1)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(probs1)

        计算方差。

        参数：
            - **probs1** (Tensor) - 伯努利实验成功的概率。默认值：None。

        返回：
            Tensor，概率分布的方差。
