mindspore.nn.probability.distribution.Gamma
================================================

.. py:class:: mindspore.nn.probability.distribution.Gamma(concentration=None, rate=None, seed=None, dtype=mindspore.float32, name='Gamma')

    伽马分布（Gamma distribution）。

    **参数：**

    - **concentration** (list, numpy.ndarray, Tensor) - 浓度，也被称为伽马分布的alpha。默认值：None。
    - **rate** (list, numpy.ndarray, Tensor) - 逆尺度参数，也被称为伽马分布的beta。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Gamma'。

    **支持平台：**

    ``Ascend``

    .. note:: 
        - `concentration` 和 `rate` 必须大于零。
        - `dtype` 必须是float，因为伽马分布是连续的。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化concentration为3.0和rate为4.0的伽马分布。
    >>> g1 = msd.Gamma([3.0], [4.0], dtype=mindspore.float32)
    >>> # 伽马分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`concentration`和`rate`必须通过参数传入。
    >>> g2 = msd.Gamma(dtype=mindspore.float32)
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
    >>> concentration_a = Tensor([2.0], dtype=mindspore.float32)
    >>> rate_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
    >>> concentration_b = Tensor([1.0], dtype=mindspore.float32)
    >>> rate_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
    >>>
    >>> # 公共接口对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，具有以下相同的参数。
    >>> # 参数：
    >>> #     value (Tensor)：要评估的值。
    >>> #     concentration (Tensor)：分布的浓度。默认值：self._concentration。
    >>> #     rate (Tensor)：分布的逆尺度。默认值：self._rate。
    >>> # `prob`示例。
    >>> # 通过将'prob'替换为函数的名称，可以对其他概率函数进行类似的调用
    >>> ans = g1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 根据分布b进行评估。
    >>> ans = g1.prob(value, concentration_b, rate_b)
    >>> print(ans.shape)
    (3,)
    >>> # 在g2的函数调用期间必须传入`concentration`和`rate`。
    >>> ans = g2.prob(value, concentration_a, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # 函数`mean`、`sd`、`mode`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     concentration (Tensor)：分布的浓度。默认值：self._concentration。
    >>> #     rate (Tensor)：分布的逆尺度。默认值：self._rate。
    >>># `mean`、`sd`、`mode`、`var`和`entropy`的示例相似。
    >>> ans = g1.mean()
    >>> print(ans.shape)
    (1,)
    >>> ans = g1.mean(concentration_b, rate_b)
    >>> print(ans.shape)
    (3,)
    >>> # 在函数调用期间必须传入`concentration`和`rate`。
    >>> ans = g2.mean(concentration_a, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # 'kl_loss'和'cross_entropy'的接口相同：
    >>> # 参数：
    >>> #     dist (str)：分布的类型。仅支持"Gamma"。
    >>> #     concentration_b (Tensor)：分布b的浓度。
    >>> #     rate_b (Tensor)：分布b的逆尺度。
    >>> #     concentration_a (Tensor)：分布a的浓度。默认值：self._concentration。
    >>> #     rate_a (Tensor)：分布a的逆尺度。默认值：self._rate。
    >>> # `kl_loss`示例。`cross_entropy`也类似。
    >>> ans = g1.kl_loss('Gamma', concentration_b, rate_b)
    >>> print(ans.shape)
    (3,)
    >>> ans = g1.kl_loss('Gamma', concentration_b, rate_b, concentration_a, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # 必须传入额外的`concentration`和`rate`。
    >>> ans = g2.kl_loss('Gamma', concentration_b, rate_b, concentration_a, rate_a)
    >>> print(ans.shape)
    (3,)
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()
    >>> #     concentration (Tensor)：分布的浓度。默认值：self._concentration。
    >>> #     rate (Tensor)：分布的逆尺度。默认值：self._rate。
    >>> ans = g1.sample()
    >>> print(ans.shape)
    (1,)
    >>> ans = g1.sample((2,3))
    >>> print(ans.shape)
    (2, 3, 1)
    >>> ans = g1.sample((2,3), concentration_b, rate_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = g2.sample((2,3), concentration_a, rate_a)
    >>> print(ans.shape)
    (2, 3, 3)
    
    .. py:method:: concentration
        :property:

        返回分布的浓度（也称为伽马分布的alpha）。
        
    .. py:method:: rate
        :property:

        返回分布的逆尺度（也称为伽马分布的beta）。
        
