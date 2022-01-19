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

    .. note:: 
        `probs` 中元素必须是合适的概率（0<p<1）。

    **异常：**

    - **ValueError** - `probs` 中元素小于0或大于1。
    - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: probs

        返回结果为1的概率。

        **返回：**

        Tensor, 结果为1的概率。
