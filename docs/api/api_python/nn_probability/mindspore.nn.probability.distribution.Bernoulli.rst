mindspore.nn.probability.distribution.Bernoulli
================================================

.. py:class:: mindspore.nn.probability.distribution.Bernoulli(probs=None, seed=None, dtype=mstype.int32, name='Bernoulli')

    伯努利分布（Bernoulli Distribution）。
    离散随机分布，取值范围为 :math:`\{0, 1\}` ，概率质量函数为 :math:`P(X = 0) = p, P(X = 1) = 1-p`。

    **参数：**

    - **probs** (float, list, numpy.ndarray, Tensor) - 结果是1的概率。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 采样结果的数据类型。默认值：mindspore.int32.
    - **name** (str) - 分布的名称。默认值：'Bernoulli'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        `probs` 中元素必须是合适的概率（0<p<1）。

    **异常：**

    - **ValueError** - `probs` 中元素小于0或大于1。
    - **TypeError** - `dtype` 不是float的子类。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化伯努利分布，probs设置为1。
    >>> b1 = msd.Bernoulli(0.5, dtype=mindspore.int32)
    >>> # 伯努利分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`probs`必须在函数调用过程中通过参数传入。
    >>> b2 = msd.Bernoulli(dtype=mindspore.int32)
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1, 0, 1], dtype=mindspore.int32)
    >>> probs_a = Tensor([0.6], dtype=mindspore.float32)
    >>> probs_b = Tensor([0.2, 0.3, 0.4], dtype=mindspore.float32)
    >>>
    >>> # 公共接口对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，它们具有相同的参数，如下所示。
    >>> # 参数：
    >>> #     value (Tensor)：要评估的值。
    >>> #     probs1 (Tensor)：成功的概率。默认值：self.probs.
    >>> # 下面是调用`prob`的示例（通过将`prob`替换为函数的名称，可以对其他概率函数进行类似的调用）：
    >>> ans = b1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 评估关于分布b的`prob`。
    >>> ans = b1.prob(value, probs_b)
    >>> print(ans.shape)
    (3,)
    >>> # `probs`必须在函数调用期间传入。
    >>> ans = b2.prob(value, probs_a)
    >>> print(ans.shape)
    (3,)
    >>>
    >>> # 函数`mean`、`sd`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     probs1 (Tensor)：成功的概率。默认值：self.probs.
    >>> # 下面是调用`mean的`示例。`sd`、`var`和`entropy`与`mean`类似。
    >>> ans = b1.mean() # return 0.5
    >>> print(ans.shape)
    ()
    >>> ans = b1.mean(probs_b) # return probs_b
    >>> print(ans.shape)
    (3,)
    >>> # `probs`必须在函数调用期间传入。
    >>> ans = b2.mean(probs_a)
    >>> print(ans.shape)
    (1,)
    >>>
    >>> # `kl_loss`和`cross_entropy`的接口如下：
    >>> # 参数：
    >>> #     dist (str)：分布的名称。仅支持'Bernoulli'。
    >>> #     probs1_b (Tensor)：分布b成功的概率。
    >>> #     probs1_a (Tensor)：分布a成功的概率。默认值：self.probs.
    >>> # 下面是调用kl_loss的示例。`cross_entropy`也类似。
    >>> ans = b1.kl_loss('Bernoulli', probs_b)
    >>> print(ans.shape)
    (3,)
    >>> ans = b1.kl_loss('Bernoulli', probs_b, probs_a)
    >>> print(ans.shape)
    (3,)
    >>> # 必须传入额外的`probs_a`。
    >>> ans = b2.kl_loss('Bernoulli', probs_b, probs_a)
    >>> print(ans.shape)
    (3,)
    >>>
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()。
    >>> #     probs1 (Tensor)：成功的概率。默认值：self.probs.
    >>> ans = b1.sample()
    >>> print(ans.shape)
    ()
    >>> ans = b1.sample((2,3))
    >>> print(ans.shape)
    (2, 3)
    >>> ans = b1.sample((2,3), probs_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = b2.sample((2,3), probs_a)
    >>> print(ans.shape)
    (2, 3, 1)
    

    .. py:method:: probs

        返回结果为1的概率。

        **返回：**

        Tensor, 结果为1的概率。
