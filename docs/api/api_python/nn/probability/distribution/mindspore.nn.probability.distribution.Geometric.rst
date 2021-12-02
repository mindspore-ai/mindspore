mindspore.nn.probability.distribution.Geometric
================================================

.. py:class:: mindspore.nn.probability.distribution.Geometric(probs=None, seed=None, dtype=mindspore.int32, name='Geometric')

    几何分布（Geometric Distribution）。

    它代表在第一次成功之前有k次失败，即在第一次成功实现时，总共有k+1个伯努利试验。

    **参数：**

    - **probs** (float, list, numpy.ndarray, Tensor) - 成功的概率。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.int32.
    - **name** (str) - 分布的名称。默认值：'Geometric'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        `probs` 必须是合适的概率（0<p<1）。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化probability为0.5的几何分布。
    >>> g1 = msd.Geometric(0.5, dtype=mindspore.int32)
    >>> # 几何分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`probs`必须在函数调用过程中通过参数传入。
    >>> g2 = msd.Geometric(dtype=mindspore.int32)
    >>>
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1, 0, 1], dtype=mindspore.int32)
    >>> probs_a = Tensor([0.6], dtype=mindspore.float32)
    >>> probs_b = Tensor([0.2, 0.5, 0.4], dtype=mindspore.float32)
    >>>
    >>> # 公共接口对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，具有以下相同的参数。
    >>> # 参数：
    >>> #     value  (Tensor)：要评估的值。
    >>> #     probs1 (Tensor)：伯努利试验成功的概率。默认值：self.probs.
    >>> # `prob`示例。
    >>> # 通过将`prob`替换为函数的名称，可以对其他概率函数进行类似的调用。
    >>> ans = g1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 根据分布b进行评估。
    >>> ans = g1.prob(value, probs_b)
    >>> print(ans.shape)
    (3,)
    >>> # `probs`必须在函数调用期间传入。
    >>> ans = g2.prob(value, probs_a)
    >>> print(ans.shape)
    (3,)
    >>> # 函数`mean`、`sd`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     probs1 (Tensor)：伯努利试验成功的概率。默认值：self.probs.
    >>> # `mean`示例。`sd`、`var`和`entropy`是相似的。
    >>> ans = g1.mean() # return 1.0
    >>> print(ans.shape)
    ()
    >>> ans = g1.mean(probs_b)
    >>> print(ans.shape)
    (3,)
    >>> # 函数调用时必须传入probs。
    >>> ans = g2.mean(probs_a)
    >>> print(ans.shape)
    (1,)
    >>> # 'kl_loss'和'cross_entropy'的接口相同。
    >>> # 参数：
    >>> #     dist (str)：分布的名称。仅支持'Geometric'。
    >>> #     probs1_b (Tensor)：伯努利分布b试验成功的概率。
    >>> #     probs1_a (Tensor)：伯努利分布a试验成功的概率。
    >>> # `kl_loss`示例。`cross_entropy`也类似。
    >>> ans = g1.kl_loss('Geometric', probs_b)
    >>> print(ans.shape)
    (3,)
    >>> ans = g1.kl_loss('Geometric', probs_b, probs_a)
    >>> print(ans.shape)
    (3,)
    >>> # 必须传入额外的`probs`。
    >>> ans = g2.kl_loss('Geometric', probs_b, probs_a)
    >>> print(ans.shape)
    (3,)
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tupler)：样本的shape。默认值：()
    >>> #     probs1 (Tensor)：伯努利试验成功的概率。默认值：self.probs.
    >>> ans = g1.sample()
    >>> print(ans.shape)
    ()
    >>> ans = g1.sample((2,3))
    >>> print(ans.shape)
    (2, 3)
    >>> ans = g1.sample((2,3), probs_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = g2.sample((2,3), probs_a)
    >>> print(ans.shape)
    (2, 3, 1)
    
    .. py:method:: probs
        :property:

        返回伯努利试验成功的概率。
        
