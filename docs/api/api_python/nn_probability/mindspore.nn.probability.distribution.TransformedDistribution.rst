mindspore.nn.probability.distribution.TransformedDistribution
==============================================================

.. py:class:: mindspore.nn.probability.distribution.TransformedDistribution(bijector, distribution, seed=None, name='transformed_distribution')

    转换分布（Transformed Distribution）。
    该类包含一个Bijector和一个分布，并通过Bijector定义的操作将原始分布转换为新分布。可如果原始分布为 :math:`X` ，Bijector的映射函数为 :math:`g(x)`，那么对应的转换分布为 :math:`Y = g(X)` 。


    参数：
        - **bijector** (Bijector) - 要执行的转换。
        - **distribution** (Distribution) - 原始分布。必须具有float数据类型。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。如果在初始化TransformedDistribution对象时给出了此种子，则对象的采样函数将使用此种子；否则，将使用基础分布的种子。
        - **name** (str) - 转换分布的名称。默认值：'transformed_distribution'。

    .. note:: 
        用于初始化原始分布的参数不能为None。例如，由于未指定 `mean` 和 `sd` ，因此无法使用mynormal = msd.Normal(dtype=mindspore.float32)初始化TransformedDistribution。
        `batch_shape` 为原始分布的 batch 的 shape。
        `broadcast_shape` 为原始分布和概率映射经过广播后的 shape。
        `is_scalar_batch` 为 True 当且仅当原始分布和概率映射的 batch 均为标量。
        `default_parameters`、 `parameter_names` 和 `parameter_type` 由原始分布的数据类型决定。
        衍生类可以通过调用 `reset_parameters` 后再调用 `add_parameter` 来添加参数以覆盖 `default_parameters` 和 `parameter_names` 。

    异常：
        - **TypeError** - bijector不是Bijector类。
        - **TypeError** - distribution不是Distribution类。

    .. py:method:: bijector
        :property:

        返回概率映射函数。

        返回：
            Bijector，概率映射函数。

    .. py:method:: distribution
        :property:

        返回变化前的概率分布。

        返回：
            Distribution，变化前的概率分布。

    .. py:method:: dtype
        :property:

        返回分布的数据类型。

        返回：
            mindspore.dtype，分布的数据类型。

    .. py:method:: is_linear_transformation
        :property:

        返回概率映射函数是否为线性映射。

        返回：
            Bool，概率映射函数为线性映射则返回True，否则返回False。

    .. py:method:: cdf(value)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: log_cdf(value)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean

        计算期望。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: prob(value)

        计算给定值下的概率。

        参数：
            - **value** (Tensor) - 要计算的值。

        返回：
            Tensor，概率值。

    .. py:method:: sample(shape)

        采样函数。

        参数：
            - **shape** (tuple) - 样本的shape。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: survival_function(value)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。

        返回：
            Tensor，生存函数的值。
