mindspore.nn.probability.distribution.Gamma
================================================

.. py:class:: mindspore.nn.probability.distribution.Gamma(concentration=None, rate=None, seed=None, dtype=mstype.float32, name='Gamma')

    伽马分布（Gamma distribution）。
    连续随机分布，取值范围为 :math:`(0, \inf)` ，概率密度函数为

    .. math::
        f(x, \alpha, \beta) = \beta^\alpha / \Gamma(\alpha) x^{\alpha - 1} \exp(-\beta x).

    其中 :math:`G` 为 Gamma 函数。

    **参数：**

    - **concentration** (int, float, list, numpy.ndarray, Tensor) - 浓度，也被称为伽马分布的alpha。默认值：None。
    - **rate** (int, float, list, numpy.ndarray, Tensor) - 逆尺度参数，也被称为伽马分布的beta。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Gamma'。

    .. note:: 
        - `concentration` 和 `rate` 中的元素必须大于零。
        - `dtype` 必须是float，因为伽马分布是连续的。

    **异常：**

    - **ValueError** - `concentration` 或者 `rate` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: concentration
        :property:

        返回分布的浓度（也称为伽马分布的alpha）。

        **返回：**

        Tensor, concentration 的值。

    .. py:method:: rate
        :property:

        返回分布的逆尺度（也称为伽马分布的beta）。

        **返回：**

        Tensor, rate 的值。

