mindspore.nn.probability.distribution.Exponential
===================================================

.. py:class:: mindspore.nn.probability.distribution.Exponential(rate=None, seed=None, dtype=mstype.float32, name='Exponential')

    指数分布（Exponential Distribution）。
    连续随机分布，取值范围为所有实数，概率密度函数为

    .. math::
        f(x, \lambda) = \lambda \exp(-\lambda x).

    其中 :math:`\lambda` 为分别为指数分布的率参数。

    **参数：**

    - **rate** (int, float, list, numpy.ndarray, Tensor) - 率参数。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Exponential'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        - `rate` 中的元素必须大于0。
        - `dtype` 必须是float，因为指数分布是连续的。

    **异常：**

    - **ValueError** - `rate` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化rate为0.5的指数分布。
    >>> e1 = msd.Exponential(0.5, dtype=mindspore.float32)
    >>> # 指数分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`rate`必须在函数调用期间通过`args`传入。
    >>> e2 = msd.Exponential(dtype=mindspore.float32)
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1, 2, 3], dtype=mindspore.float32)
    >>> rate_a = Tensor([0.6], dtype=mindspore.float32)
    >>> rate_b = Tensor([0.2, 0.5, 0.4], dtype=mindspore.float32)
    >>> # 公共接口对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，如下所示。
    >>> # 参数：
    >>> #     value (Tensor)：要评估的值。
    >>> #     rate (Tensor)：分布的率参数。默认值：self.rate.
    >>> # `prob`示例。
    >>> # 通过将`prob`替换为函数的名称，可以对其他概率函数进行类似的调用。
    >>> ans = e1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 根据分布b进行评估。
    >>> ans = e1.prob(value, rate_b)
    >>> print(ans.shape)
    (3,)
    >>> # `rate`必须在函数调用期间传入。
    >>> ans = e2.prob(value, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # 函数`mean`、`sd`、`var`和`entropy`具有相同的参数，如下所示。
    >>> # 参数：
    >>> #     rate (Tensor)：分布的率参数。默认值：self.rate.
    >>> # `mean`示例。`sd`、`var`和`entropy`是相似的。
    >>> ans = e1.mean() # return 2
    >>> print(ans.shape)
    ()
    >>> ans = e1.mean(rate_b) # return 1 / rate_b
    >>> print(ans.shape)
    (3,)
    >>> # `rate`必须在函数调用期间传入。
    >>> ans = e2.mean(rate_a)
    >>> print(ans.shape)
    (1,)
    >>> # `kl_loss`和`cross_entropy`的接口相同。
    >>> # 参数：
    >>> #     dist (str)：分布的名称。仅支持'Exponential'。
    >>> #     rate_b (Tensor)：分布b的率参数。
    >>> #     rate_a (Tensor)：分布a的率参数。默认值：self.rate.
    >>> # `kl_loss`示例。`cross_entropy`也类似。
    >>> ans = e1.kl_loss('Exponential', rate_b)
    >>> print(ans.shape)
    (3,)
    >>> ans = e1.kl_loss('Exponential', rate_b, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # 必须传入额外的`rate`。
    >>> ans = e2.kl_loss('Exponential', rate_b, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()
    >>> #     probs1 (Tensor)：分布的率参数。默认值：self.rate.
    >>> ans = e1.sample()
    >>> print(ans.shape)
    ()
    >>> ans = e1.sample((2,3))
    >>> print(ans.shape)
    (2, 3)
    >>> ans = e1.sample((2,3), rate_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = e2.sample((2,3), rate_a)
    >>> print(ans.shape)
    (2, 3, 1)

    .. py:method:: rate
        :property:

        返回 `rate` 。

        **返回：**

        Tensor, rate 的值。

