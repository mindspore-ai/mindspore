mindspore.nn.probability.distribution.Geometric
================================================

.. py:class:: mindspore.nn.probability.distribution.Geometric(probs=None, seed=None, dtype=mstype.int32, name='Geometric')

    几何分布（Geometric Distribution）。

    它代表在第一次成功之前有k次失败，即在第一次成功实现时，总共有k+1个伯努利试验。
    离散随机分布，取值范围为正自然数集，概率质量函数为 :math:`P(X = i) = p(1-p)^{i-1}, i = 1, 2, ...`。

    **参数：**

    - **probs** (int, float, list, numpy.ndarray, Tensor) - 成功的概率。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.int32.
    - **name** (str) - 分布的名称。默认值：'Geometric'。

    .. note:: 
        `probs` 必须是合适的概率（0<p<1）。


    **异常：**

    - **ValueError** - `probs` 中元素小于0或者大于1。

    .. py:method:: probs
        :property:

        返回伯努利试验成功的概率。

        **返回：**

        Tensor, 伯努利试验成功的概率值。

