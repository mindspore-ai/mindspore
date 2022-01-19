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

    .. note:: 
        `probs` 的秩必须至少为1，值是合适的概率，并且总和为1。

    **异常：**

    - **ValueError** - `probs` 的秩为0或者其中所有元素的和不等于1。

    .. py:method:: probs

        返回事件发生的概率。

        **返回：**

        Tensor, 事件发生的概率。

