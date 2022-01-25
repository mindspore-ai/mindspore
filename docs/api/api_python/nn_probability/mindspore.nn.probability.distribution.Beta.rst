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

    .. note::
        - `concentration1` 和 `concentration0` 中元素必须大于零。
        - `dtype` 必须是float，因为 Beta 分布是连续的。

    **异常：**

    - **ValueError** - `concentration1` 或者 `concentration0` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

      返回concentration0（也称为 Beta 分布的beta）。

    **返回：**

    Tensor, concentration0 的值。

    .. py:method:: concentration1
        :property:

        返回concentration1（也称为 Beta 分布的alpha）。

        **返回：**

        Tensor, concentration1 的值。

