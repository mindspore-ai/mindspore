mindspore.nn.probability.distribution.Exponential
===================================================

.. py:class:: mindspore.nn.probability.distribution.Exponential(rate=None, seed=None, dtype=mstype.float32, name='Exponential')

    指数分布（Exponential Distribution）。
    连续随机分布，取值范围为所有实数，概率密度函数为

    .. math::
        f(x, \lambda) = \lambda \exp(-\lambda x).

    其中 :math:`\lambda` 为分别为指数分布的率参数。

    **参数：**

    - **rate** (int, float, list, numpy.ndarray, Tensor) - 率参数。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Exponential'。

    .. note:: 
        - `rate` 中的元素必须大于0。
        - `dtype` 必须是float，因为指数分布是连续的。

    **异常：**

    - **ValueError** - `rate` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: rate
        :property:

        返回 `rate` 。

        **返回：**

        Tensor, rate 的值。

