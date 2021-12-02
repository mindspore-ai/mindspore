mindspore.nn.probability.distribution.Uniform
================================================

.. py:class:: mindspore.nn.probability.distribution.Uniform(low=None, high=None, seed=None, dtype=mindspore.float32, name='Uniform')

    示例类：均匀分布（Uniform Distribution）。

    **参数：**

    - **low** (int, float, list, numpy.ndarray, Tensor) - 分布的下限。默认值：None。
    - **high** (int, float, list, numpy.ndarray, Tensor) - 分布的上限。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Uniform'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        - `low` 必须小于 `high` 。
        - `dtype` 必须是float类型，因为均匀分布是连续的。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化lower bound为0.0和higher bound为1.0的均匀分布。
    >>> u1 = msd.Uniform(0.0, 1.0, dtype=mindspore.float32)
    >>> # 平均分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`high`和`low`必须在函数调用过程中通过参数传入。
    >>> u2 = msd.Uniform(dtype=mindspore.float32)
    >>>
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([0.5, 0.8], dtype=mindspore.float32)
    >>> low_a = Tensor([0., 0.], dtype=mindspore.float32)
    >>> high_a = Tensor([2.0, 4.0], dtype=mindspore.float32)
    >>> low_b = Tensor([-1.5], dtype=mindspore.float32)
    >>> high_b = Tensor([2.5, 5.], dtype=mindspore.float32)
    >>> # 公共接口对应的概率函数的私有接口，包括
    >>> # `prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，具有相同的参数。
    >>> # 参数：
    >>> #     value (Tensor)：要评估的值。
    >>> #     low (Tensor)：分布的下限。默认值：self.low.
    >>> #     high (Tensor)：分布的上限。默认值：self.high。
    >>> # `prob`示例。
    >>> # 通过将'prob'替换为函数的名称，可以对其他概率函数进行类似的调用。        
    >>> ans = u1.prob(value)
    >>> print(ans.shape)
    (2,)
    >>> # 根据分布b进行评估。
    >>> ans = u1.prob(value, low_b, high_b)
    >>> print(ans.shape)
    (2,)
    >>> # `high`和`low`必须在函数调用期间传入。
    >>> ans = u2.prob(value, low_a, high_a)
    >>> print(ans.shape)
    (2,)
    >>> # 函数`mean`、`sd`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     low (Tensor)：分布的下限。默认值：self.low.
    >>> #     high (Tensor)：分布的上限。默认值：self.high。
    >>> # `mean`示例。`sd`、`var`和`entropy`是相似的。
    >>> ans = u1.mean() # return 0.5
    >>> print(ans.shape)
    ()
    >>> ans = u1.mean(low_b, high_b) # return (low_b + high_b) / 2
    >>> print(ans.shape)
    (2,)
    >>> # `high`和`low`必须在函数调用期间传入。
    >>> ans = u2.mean(low_a, high_a)
    >>> print(ans.shape)
    (2,)
    >>> # 'kl_loss'和'cross_entropy'的接口相同。
    >>> # 参数：
    >>> #     dist (str)：分布的类型。在这种情况下，应该是"Uniform"。
    >>> #     low_b (Tensor)：分布b的下限。
    >>> #     high_b (Tensor)：分布b的上限。
    >>> #     low_a (Tensor)：分布a的下限。默认值：self.low.
    >>> #     high_a (Tensor)：分布a的上限。默认值：self.high。
    >>> # `kl_loss`示例。`cross_entropy`也类似。
    >>> ans = u1.kl_loss('Uniform', low_b, high_b)
    >>> print(ans.shape)
    (2,)
    >>> ans = u1.kl_loss('Uniform', low_b, high_b, low_a, high_a)
    >>> print(ans.shape)
    (2,)
    >>> # 必须传入额外的`high`和`low`。
    >>> ans = u2.kl_loss('Uniform', low_b, high_b, low_a, high_a)
    >>> print(ans.shape)
    (2,)
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()
    >>> #     low (Tensor)：分布的下限。默认值：self.low.
    >>> #     high (Tensor)：分布的上限。默认值：self.high。
    >>> ans = u1.sample()
    >>> print(ans.shape)
    ()
    >>> ans = u1.sample((2,3))
    >>> print(ans.shape)
    (2, 3)
    >>> ans = u1.sample((2,3), low_b, high_b)
    >>> print(ans.shape)
    (2, 3, 2)
    >>> ans = u2.sample((2,3), low_a, high_a)
    >>> print(ans.shape)
    (2, 3, 2)
    
    .. py:method:: high
        :property:

        返回分布的上限。
        
    .. py:method:: low
        :property:

        返回分布的下限。
        