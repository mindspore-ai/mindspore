mindspore.nn.probability.distribution.Poisson
================================================

.. py:class:: mindspore.nn.probability.distribution.Poisson(rate=None, seed=None, dtype=mstype.float32, name='Poisson')

    泊松分布（Poisson Distribution）。
    离散随机分布，取值范围为正自然数集，概率质量函数为

    .. math::
        P(X = k) = \lambda^k \exp(-\lambda) / k!, k = 1, 2, ...

    其中 :math:`\lambda` 为率参数（rate)。

    **参数：**

    - **rate** (int, float, list, numpy.ndarray, Tensor) - 泊松分布的率参数。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Poisson'。

    .. note:: 
        `rate` 必须大于0。


    **异常：**

    - **ValueError** - `rate` 中元素小于0。

    .. py:method:: rate
        :property:

        返回分布的 `rate` 参数。

        **返回：**

        Tensor, rate 参数的值。

