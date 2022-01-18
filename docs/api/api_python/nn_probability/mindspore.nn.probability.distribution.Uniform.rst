mindspore.nn.probability.distribution.Uniform
================================================

.. py:class:: mindspore.nn.probability.distribution.Uniform(low=None, high=None, seed=None, dtype=mstype.float32, name='Uniform')

    均匀分布（Uniform Distribution）。
    连续随机分布，取值范围为 :math:`[a, b]` ，概率密度函数为

    .. math:: 
        f(x, a, b) = 1 / (b - a).

    其中 :math:`a, b` 为分别为均匀分布的下界和上界。

    **参数：**

    - **low** (int, float, list, numpy.ndarray, Tensor) - 分布的下限。默认值：None。
    - **high** (int, float, list, numpy.ndarray, Tensor) - 分布的上限。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Uniform'。

    .. note:: 
        - `low` 必须小于 `high` 。
        - `dtype` 必须是float类型，因为均匀分布是连续的。


    **异常：**

    - **ValueError** - `low` 大于等于 `high` 。
    - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: high
        :property:

        返回分布的上限。

        **返回：**

        Tensor, 分布的上限值。

    .. py:method:: low
        :property:

        返回分布的下限。

        **返回：**

        Tensor, 分布的下限值。