mindspore.nn.probability.distribution.Normal
================================================

.. py:class:: mindspore.nn.probability.distribution.Normal(mean=None, sd=None, seed=None, dtype=mstype.float32, name='Normal')

    正态分布（Normal distribution）。
    连续随机分布，取值范围为 :math:`(-\inf, \inf)` ，概率密度函数为

    .. math:: 
        f(x, \mu, \sigma) = 1 / \sigma\sqrt{2\pi} \exp(-(x - \mu)^2 / 2\sigma^2).

    其中 :math:`\mu, \sigma` 为分别为正态分布的期望与标准差。

    **参数：**

    - **mean** (int, float, list, numpy.ndarray, Tensor) - 正态分布的平均值。默认值：None。
    - **sd** (int, float, list, numpy.ndarray, Tensor) - 正态分布的标准差。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Normal'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        - `sd` 必须大于零。
        - `dtype` 必须是float，因为正态分布是连续的。

    **异常：**

    - **ValueError** - `sd` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化mean为3.0和standard deviation为4.0的正态分布。
    >>> n1 = msd.Normal(3.0, 4.0, dtype=mindspore.float32)
    >>> # 正态分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`mean`和`sd`必须通过参数传入。
    >>> n2 = msd.Normal(dtype=mindspore.float32)
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
    >>> mean_a = Tensor([2.0], dtype=mindspore.float32)
    >>> sd_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
    >>> mean_b = Tensor([1.0], dtype=mindspore.float32)
    >>> sd_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
    >>> # 公共接口对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，具有以下相同的参数。
    >>> # 参数：
    >>> #     value (Tensor)：要评估的值。
    >>> #     mean (Tensor)：分布的均值。默认值：self._mean_value。
    >>> #     sd (Tensor)：分布的标准差。默认值：self._sd_value。
    >>> # `prob`示例。
    >>> # 通过将'prob'替换为其他概率函数的名称，可以对其他概率函数进行类似的调用
    >>> ans = n1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 根据分布b进行评估。
    >>> ans = n1.prob(value, mean_b, sd_b)
    >>> print(ans.shape)
    (3,)
    >>> # 在函数调用期间必须传入`mean`和`sd`
    >>> ans = n2.prob(value, mean_a, sd_a)
    >>> print(ans.shape)
    (3,)
    >>> # 函数`mean`、`sd`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     mean (Tensor)：分布的均值。默认值：self._mean_value。
    >>> #     sd (Tensor)：分布的标准差。默认值：self._sd_value。
    >>> # 'mean'示例。`sd`、`var`和`entropy`是相似的。
    >>> ans = n1.mean() # return 0.0
    >>> print(ans.shape)
    ()
    >>> ans = n1.mean(mean_b, sd_b) # return mean_b
    >>> print(ans.shape)
    (3,)
    >>> # `mean`和`sd`必须在函数调用期间传入。
    >>> ans = n2.mean(mean_a, sd_a)
    >>> print(ans.shape)
    (3,)
    >>> # 'kl_loss'和'cross_entropy'的接口相同：
    >>> # 参数：
    >>> #     dist (str)：分布的类型。仅支持"Normal"。
    >>> #     mean_b (Tensor)：分布b的均值。
    >>> #     sd_b（Tensor)：分布b的标准差。
    >>> #     mean_a (Tensor)：分布a的均值。默认值：self._mean_value。
    >>> #     sd_a（Tensor)：分布a的标准差。默认值：self._sd_value。
    >>> # `kl_loss`示例。`cross_entropy`也类似。
    >>> ans = n1.kl_loss('Normal', mean_b, sd_b)
    >>> print(ans.shape)
    (3,)
    >>> ans = n1.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
    >>> print(ans.shape)
    (3,)
    >>> # 必须传入额外的`mean`和`sd`。
    >>> ans = n2.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
    >>> print(ans.shape)
    (3,)
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()
    >>> #     mean (Tensor)：分布的均值。默认值：self._mean_value。
    >>> #     sd (Tensor)：分布的标准差。默认值：self._sd_value。
    >>> ans = n1.sample()
    >>> print(ans.shape)
    ()
    >>> ans = n1.sample((2,3))
    >>> print(ans.shape)
    (2, 3)
    >>> ans = n1.sample((2,3), mean_b, sd_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = n2.sample((2,3), mean_a, sd_a)
    >>> print(ans.shape)
    (2, 3, 3)

