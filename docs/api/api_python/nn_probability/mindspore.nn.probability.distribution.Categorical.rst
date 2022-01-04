mindspore.nn.probability.distribution.Categorical
==================================================

.. py:class:: mindspore.nn.probability.distribution.Categorical(probs=None, seed=None, dtype=mstype.float32, name='Categorical')

    分类分布。
    离散随机分布，取值范围为 :math:`\{1, 2, ..., k\}` ，概率质量函数为 :math:`P(X = i) = p_i, i = 1, ..., k`。

    **参数：**

    - **probs** (Tensor, list, numpy.ndarray) - 事件概率。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.int32.
    - **name** (str) - 分布的名称。默认值：Categorical。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        `probs` 的秩必须至少为1，值是合适的概率，并且总和为1。

    **异常：**

    - **ValueError** - `probs` 的秩为0或者其中所有元素的和不等于1。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> # 初始化probs为[0.2, 0.8]的类别分布。
    >>> ca1 = msd.Categorical(probs=[0.2, 0.8], dtype=mindspore.int32)
    >>> # 类别分布可以在没有参数的情况下初始化。
    >>> # 在这种情况下，`probs`必须在函数调用过程中通过参数传入。
    >>> ca2 = msd.Categorical(dtype=mindspore.int32)
    >>> # 下面是用于测试的Tensor
    >>> value = Tensor([1, 0], dtype=mindspore.int32)
    >>> probs_a = Tensor([0.5, 0.5], dtype=mindspore.float32)
    >>> probs_b = Tensor([0.35, 0.65], dtype=mindspore.float32)
    >>> # 公共接口所对应的概率函数的私有接口，包括`prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`，如下所示。
    >>> # 参数：
    >>> #     value (Tensor)：要评估的值。
    >>> #     probs (Tensor)：事件概率。默认值：self.probs.
    >>> # `prob`示例。
    >>> # 通过将`prob`替换为函数的名称，可以对其他概率函数进行类似的调用。
    >>> ans = ca1.prob(value)
    >>> print(ans.shape)
    (2,)
    >>> # 评估关于分布b的`prob`。
    >>> ans = ca1.prob(value, probs_b)
    >>> print(ans.shape)
    (2,)
    >>> # `probs`必须在函数调用期间传入。
    >>> ans = ca2.prob(value, probs_a)
    >>> print(ans.shape)
    (2,)
    >>> # 函数`mean`、`sd`、`var`和`entropy`具有相同的参数。
    >>> # 参数：
    >>> #     probs (Tensor)：事件概率。默认值：self.probs.
    >>> # `mean`示例。`sd`、`var`和`entropy`是相似的。
    >>> ans = ca1.mean() # return 0.8
    >>> print(ans.shape)
    (1,)
    >>> ans = ca1.mean(probs_b)
    >>> print(ans.shape)
    (1,)
    >>> # `probs`必须在函数调用期间传入。
    >>> ans = ca2.mean(probs_a)
    >>> print(ans.shape)
    (1,)
    >>> # `kl_loss`和`cross_entropy`的接口如下：
    >>> # 参数：
    >>> #     dist (str)：分布的名称。仅支持'Categorical'。
    >>> #     probs_b (Tensor)：分布b的事件概率。
    >>> #     probs (Tensor)：分布a的事件概率。默认值：self.probs.
    >>> # kl_loss示例。`cross_entropy`也类似。
    >>> ans = ca1.kl_loss('Categorical', probs_b)
    >>> print(ans.shape)
    ()
    >>> ans = ca1.kl_loss('Categorical', probs_b, probs_a)
    >>> print(ans.shape)
    ()
    >>> # 必须传入额外的`probs`。
    >>> ans = ca2.kl_loss('Categorical', probs_b, probs_a)
    >>> print(ans.shape)
    ()

    .. py:method:: probs

        返回事件发生的概率。

        **返回：**

        Tensor, 事件发生的概率。

