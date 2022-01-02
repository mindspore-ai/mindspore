mindspore.nn.probability.distribution.Beta
================================================

.. py:class:: mindspore.nn.probability.distribution.Beta(concentration1=None, concentration0=None, seed=None, dtype=mstype.float32, name='Beta')

    Beta 分布（Beta Distribution）。
    连续随机分布，取值范围为 :math:`[0, 1]` ，概率密度函数为 

    .. math:: 
        f(x, \alpha, \beta) = x^\alpha (1-x)^{\beta - 1} / B(\alpha, \beta).

    其中 :math:`B` 为 Beta 函数。

    **参数：**

    - **concentration1** (int, float, list, numpy.ndarray, Tensor) - Beta 分布的alpha。
    - **concentration0** (int, float, list, numpy.ndarray, Tensor) - Beta 分布的beta。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 采样结果的数据类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Beta'。

    **支持平台：**

    ``Ascend``

    .. note:: 
        - `concentration1` 和 `concentration0` 中元素必须大于零。
        - `dtype` 必须是float，因为 Beta 分布是连续的。

    **异常：**

    - **ValueError** - `concentration1` 或者 `concentration0` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>>
    >>> # 初始化concentration1为3.0和concentration0为4.0的 Beta 分布。
    >>> b1 = msd.Beta([3.0], [4.0], dtype=mindspore.float32)
    >>>
    >>> # Beta分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`concentration1`和`concentration0`必须通过参数传入。
    >>> b2 = msd.Beta(dtype=mindspore.float32)
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([0.1, 0.5, 0.8], dtype=mindspore.float32)
    >>> c1_a = Tensor([2.0], dtype=mindspore.float32)
    >>> c0_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
    >>> c1_b = Tensor([1.0], dtype=mindspore.float32)
    >>> c0_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
    >>>
    >>> # 公共接口对应的概率函数的私有接口（包括`prob`和`log_prob`）的参数相同，如下所示。
    >>> # 参数：
    >>> #     value (Tensor)：要评估的值。
    >>> #     concentration1 (Tensor)：分布的concentration1。默认值：self._concentration1。
    >>> #     concentration0 (Tensor)：分布的concentration0。默认值：self._concentration0。
    >>> # 下面是`prob`的示例（通过将'prob'替换为函数的名称，可以对其他概率函数进行类似的调用）：
    >>> ans = b1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 根据分布b进行评估。
    >>> ans = b1.prob(value, concentration1_b, concentration0_b)
    >>> print(ans.shape)
    (3,)
    >>> # 在函数调用期间必须传入`concentration1`和`concentration0`
    >>> ans = b2.prob(value, concentration1=c1_a, concentration0=c0_a)
    >>> print(ans.shape)
    (3,)
    >>>
    >>> # 函数`mean`、`sd`、`mode`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     concentration1 (Tensor)：分布的concentration1。默认值：self._concentration1。
    >>> #     concentration0 (Tensor)：分布的concentration0。默认值：self._concentration0。
    >>> # 下面是调研`mean`的示例（`sd`、`mode`、`var`和`entropy`的示例与`mean`相似）：
    >>> ans = b1.mean()
    >>> print(ans.shape)
    (1,)
    >>> ans = b1.mean(concentration1=c1_b, concentration0=c0_b)
    >>> print(ans.shape)
    (3,)
    >>> # `concentration1`和`concentration0`必须在函数调用期间传入。
    >>> ans = b2.mean(concentration1=c1_a, concentration0=c0_a)
    >>> print(ans.shape)
    (3,)
    >>>
    >>> # 'kl_loss'和'cross_entropy'的接口相同：
    >>> # 参数：
    >>> #     dist (str)：分布的类型。仅支持"Beta"。
    >>> #     concentration1_b (Tensor)：分布b的concentration1。
    >>> #     concentration0_b (Tensor)：分布b的concentration0。
    >>> #     concentration1_a (Tensor)：分布a的concentration1。
    >>> #       默认值：self._concentration1。
    >>> #     concentration0_a (Tensor)：分布a的concentration0。
    >>> #       默认值：self._concentration0。
    >>> # 下面是`kl_loss`示例（`cross_entropy`也类似）：
    >>> ans = b1.kl_loss('Beta', concentration1_b=c1_b, concentration0_b=c0_b)
    >>> print(ans.shape)
    (3,)
    >>> ans = b1.kl_loss('Beta', concentration1_b=c1_b, concentration0_b=c0_b, concentration1_a=c1_a, concentration0_a=c0_a)
    >>> print(ans.shape)
    (3,)
    >>>
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()
    >>> #     concentration1 (Tensor)：分布的concentration1。默认值：self._concentration1。
    >>> #     concentration0 (Tensor)：分布的concentration0。默认值：self._concentration0。
    >>> ans = b1.sample()
    >>> print(ans.shape)
    (1,)
    >>> ans = b1.sample((2,3))
    >>> print(ans.shape)
    (2, 3, 1)
    >>> ans = b1.sample((2,3), concentration1=c1_b, concentration0=c0_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = b2.sample((2,3), concentration1=c1_a, concentration0=c0_a)
    >>> print(ans.shape)
    (2, 3, 3)
    
    .. py:method:: concentration0
        :property:

        返回concentration0（也称为 Beta 分布的beta）。

        **返回：**

        Tensor, concentration0 的值。
        
    .. py:method:: concentration1
        :property:

        返回concentration1（也称为 Beta 分布的alpha）。

        **返回：**

        Tensor, concentration1 的值。

