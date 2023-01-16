mindspore.nn.probability.distribution.StudentT
================================================

.. py:class:: mindspore.nn.probability.distribution.StudentT(df=None, mean=None, sd=None, seed=None, dtype=mstype.float32, name='StudentT')

    StudentT分布（StudentT distribution）。
    连续随机分布，取值范围为 :math:`(-\inf, \inf)` ，概率密度函数为

    .. math::
        f(x, \nu, \mu, \sigma) = (1 + y^2 / \nu)^{(-0.5*(\nu + 1))} / Z


    其中 :math:`y = (x - \mu) / \sigma`,  :math:`Z = abs(\sigma) * \sqrt(\nu * \pi) * \Gamma(0.5 * \nu) / \Gamma(0.5 * (\nu + 1))`, :math:`\nu, \mu, \sigma` 为分别为StudentT分布的自由度，期望与标准差。

    参数：
        - **df** (int, float, list, numpy.ndarray, Tensor) - StudentT分布的自由度。默认值：None。
        - **mean** (int, float, list, numpy.ndarray, Tensor) - StudentT分布的平均值。默认值：None。
        - **sd** (int, float, list, numpy.ndarray, Tensor) - StudentT分布的扩散度。默认值：None。
        - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
        - **dtype** (mindspore.dtype) - 事件样例的类型。默认值：mstype.float32。
        - **name** (str) - 分布的名称。默认值：'StudentT'。

    .. note:: 
        - `df` 必须大于0。
        - `sd` 必须大于0。
        - `dtype` 必须是float，因为StudentT分布是连续的。

    异常：
        - **ValueError** - `df` 中元素不大于0。
        - **ValueError** - `sd` 中元素不大于0。
        - **TypeError** - `dtype` 不是float的子类。

    .. py:method:: log_prob(value, df=None, mean=None, sd=None)

        计算给定值对应的概率的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **df** (Tensor) - 分布的自由度。默认值：None。
            - **mean** (Tensor) - 分布的期望。默认值：None。
            - **sd** (Tensor) - 分布的扩散度。默认值：None。

        返回：
            Tensor，概率密度函数的对数。