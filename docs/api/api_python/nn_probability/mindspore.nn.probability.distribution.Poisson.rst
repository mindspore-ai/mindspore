mindspore.nn.probability.distribution.Poisson
================================================

.. py:class:: mindspore.nn.probability.distribution.Poisson(rate=None, seed=None, dtype=mstype.float32, name='Poisson')

    泊松分布（Poisson Distribution）。
    离散随机分布，取值范围为正自然数集，概率质量函数为

    .. math::
        P(X = k) = \lambda^k \exp(-\lambda) / k!, k = 1, 2, ...

    其中 :math:`\lambda` 为率参数（rate)。

    **参数：**

    - **rate** (int, float, list, numpy.ndarray, Tensor) - 泊松分布的率参数。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Poisson'。

    **支持平台：**

    ``Ascend``

    .. note:: 
        `rate` 必须大于0。


    **异常：**

    - **ValueError** - `rate` 中元素小于0。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化rate为0.5的泊松分布。
    >>> p1 = msd.Poisson([0.5], dtype=mindspore.float32)
    >>> # 泊松分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`rate`必须在函数调用期间通过`args`传入。
    >>> p2 = msd.Poisson(dtype=mindspore.float32)
    >>>
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1, 2, 3], dtype=mindspore.int32)
    >>> rate_a = Tensor([0.6], dtype=mindspore.float32)
    >>> rate_b = Tensor([0.2, 0.5, 0.4], dtype=mindspore.float32)
    >>>
    >>> # 公共接口对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，如下所示。
    >>> # 参数：
    >>> #     value  (Tensor)：要评估的值。
    >>> #     rate (Tensor)：分布的率参数。默认值：self.rate.
    >>> # `prob`示例。
    >>> # 通过将`prob`替换为其他概率函数的名称，可以对其他概率函数进行类似的调用。
    >>> ans = p1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 根据分布b进行评估。
    >>> ans = p1.prob(value, rate_b)
    >>> print(ans.shape)
    (3,)
    >>> # `rate`必须在函数调用期间传入。
    >>> ans = p2.prob(value, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # 函数`mean`、`mode`、`sd`和'var'具有相同的参数，如下所示。
    >>> # 参数：
    >>> #     rate (Tensor)：分布的率参数。默认值：self.rate.
    >>> # `mean`、`sd`、`mode`和`var`的示例都类似。
    >>> ans = p1.mean() # return 2
    >>> print(ans.shape)
    (1,)
    >>> ans = p1.mean(rate_b) # return 1 / rate_b
    >>> print(ans.shape)
    (3,)
    >>> # `rate`必须在函数调用期间传入。
    >>> ans = p2.mean(rate_a)
    >>> print(ans.shape)
    (1,)
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()
    >>> #     probs1 (Tensor)：分布的率参数。默认值：self.rate.
    >>> ans = p1.sample()
    >>> print(ans.shape)
    (1, )
    >>> ans = p1.sample((2,3))
    >>> print(ans.shape)
    (2, 3, 1)
    >>> ans = p1.sample((2,3), rate_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = p2.sample((2,3), rate_a)
    >>> print(ans.shape)
    (2, 3, 1)

    .. py:method:: rate
        :property:

        返回分布的 `rate` 参数。

        **返回：**

        Tensor, rate 参数的值。

