mindspore.nn.probability.distribution.Logistic
================================================

.. py:class:: mindspore.nn.probability.distribution.Logistic(loc=None, scale=None, seed=None, dtype=mstype.float32, name='Logistic')

    Logistic分布（Logistic distribution）。
    连续随机分布，取值范围为 :math:`(0, \inf)` ，概率密度函数为

    .. math::
        f(x, a, b) = 1 / b \exp(\exp(-(x - a) / b) - x).

    其中 :math:`a, b` 为分别为Logistic分布的位置参数和比例参数。

    **参数：**

    - **loc** (int, float, list, numpy.ndarray, Tensor) - Logistic分布的位置。默认值：None。
    - **scale** (int, float, list, numpy.ndarray, Tensor) - Logistic分布的尺度。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Logistic'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        - `scale` 必须大于零。
        - `dtype` 必须是float，因为Logistic分布是连续的。

    **异常：**

    - **ValueError** - `scale` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化loc为3.0和scale为4.0的Logistic分布。
    >>> l1 = msd.Logistic(3.0, 4.0, dtype=mindspore.float32)
    >>> # 可以在没有参数的情况下初始化Logistic分布。
    >>> # 在这种情况下，`loc`和`scale`必须通过参数传入。
    >>> l2 = msd.Logistic(dtype=mindspore.float32)
    >>>
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
    >>> loc_a = Tensor([2.0], dtype=mindspore.float32)
    >>> scale_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
    >>> loc_b = Tensor([1.0], dtype=mindspore.float32)
    >>> scale_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
    >>>
    >>> # 公共接口对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，具有以下相同的参数。
    >>> # 参数：
    >>> #     value  (Tensor)：要评估的值。
    >>> #     loc (Tensor)：分布的位置。默认值：self.loc.
    >>> #     scale (Tensor)：分布的尺度。默认值：self.scale.
    >>> # `prob`示例。
    >>> # 通过将'prob'替换为函数的名称，可以对其他概率函数进行类似的调用
    >>> ans = l1.prob(value)
    >>> print(ans.shape)
    (3,)
    >>> # 根据分布b进行评估。
    >>> ans = l1.prob(value, loc_b, scale_b)
    >>> print(ans.shape)
    (3,)
    >>> # 在函数调用期间必须传入`loc`和`scale`
    >>> ans = l1.prob(value, loc_a, scale_a)
    >>> print(ans.shape)
    (3,)
    >>> # 函数`mean`、`mode`、`sd`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     loc (Tensor)：分布的位置。默认值：self.loc.
    >>> #     scale (Tensor)：分布的尺度。默认值：self.scale.
    >>> # 'mean'示例。`mode`、`sd`、`var`和`entropy`也类似。
    >>> ans = l1.mean()
    >>> print(ans.shape)
    ()
    >>> ans = l1.mean(loc_b, scale_b)
    >>> print(ans.shape)
    (3,)
    >>> # 在函数调用期间必须传入`loc`和`scale`。
    >>> ans = l1.mean(loc_a, scale_a)
    >>> print(ans.shape)
    (3,)
    >>> # `sample`示例。
    >>> # 参数：
    >>> #     shape (tuple)：样本的shape。默认值：()
    >>> #     loc (Tensor)：分布的位置。默认值：self.loc.
    >>> #     scale (Tensor)：分布的尺度。默认值：self.scale.
    >>> ans = l1.sample()
    >>> print(ans.shape)
    ()
    >>> ans = l1.sample((2,3))
    >>> print(ans.shape)
    (2, 3)
    >>> ans = l1.sample((2,3), loc_b, scale_b)
    >>> print(ans.shape)
    (2, 3, 3)
    >>> ans = l1.sample((2,3), loc_a, scale_a)
    >>> print(ans.shape)
    (2, 3, 3)

    .. py:method:: loc
        :property:

        返回分布位置。

        **返回：**

        Tensor, 分布的位置值。

    .. py:method:: scale
        :property:

        返回分布比例。

        **返回：**

        Tensor, 分布的比例值。

