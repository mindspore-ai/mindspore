mindspore.nn.probability.distribution.TransformedDistribution
==============================================================

.. py:class:: mindspore.nn.probability.distribution.TransformedDistribution(bijector, distribution, seed=None, name='transformed_distribution')

    转换分布（Transformed Distribution）。
    该类包含一个Bijector和一个分布，并通过Bijector定义的操作将原始分布转换为新分布。可如果原始分布为 :math:`X` ，Bijector的映射函数为 :math:`g`， 那么对应的转换分布为 :math:`Y = g(X)` 。


    **参数：**

    - **bijector** (Bijector) - 要执行的转换。
    - **distribution** (Distribution) - 原始分布。必须具有float数据类型。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。如果在初始化TransformedDistribution对象时给出了此种子，则对象的采样函数将使用此种子；否则，将使用基础分布的种子。
    - **name** (str) - 转换分布的名称。默认值：'transformed_distribution'。

    .. note:: 
        用于初始化原始分布的参数不能为None。例如，由于未指定 `mean` 和 `sd` ，因此无法使用mynormal = msd.Normal(dtype=mindspore.float32)初始化TransformedDistribution。

    **异常：**

    - **TypeError** - bijector不是Bijector类。
    - **TypeError** - distribution不是Distribution类。

    .. py:method:: bijector
        :property:

        返回概率映射函数。

        **返回：**

        Bijector, 概率映射函数。

    .. py:method:: distribution
        :property:

        返回变化前的概率分布。

        **返回：**

        Distribution, 变化前的概率分布。

    .. py:method:: is_linear_transformation
        :property:

        返回概率映射函数是否为线性映射。

        **返回：**

        Bool, 概率映射函数为线性映射则返回True，否则返回False。