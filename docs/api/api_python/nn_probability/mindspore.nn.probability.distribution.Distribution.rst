mindspore.nn.probability.distribution.Distribution
===================================================

.. py:class:: mindspore.nn.probability.distribution.Distribution(seed, dtype, name, param)

    所有分布的基类。

    参数：
        - **seed** (int) - 采样时使用的种子。如果为None，则使用0。
        - **dtype** (mindspore.dtype) - 事件样例的类型。
        - **name** (str) - 分布的名称。
        - **param** (dict) - 用于初始化分布的参数。

    .. note:: 
        派生类必须重写 `_mean` 、 `_prob` 和 `_log_prob` 等操作。必填参数必须通过 `args` 或 `kwargs` 传入，如 `_prob` 的 `value` 。
        `dist_spec_args` 作为可选参数可以用来制定新的分布参数。

        每种分类都有自己的 `dist_spec_args`。例如正态分布的 `dist_spec_args` 为 `mean` 和 `sd`，
        而指数分布的 `dist_spec_args` 为 `rate`。

        所有方法都包含一个 `dist_spec_args` 作为可选参数。
        传入 `dist_spec_args` 可以让该方法基于新的分布的参数值进行运算。但如此做不会改变原始分布的参数。

    .. py:method:: cdf(value, *args, **kwargs)

        在给定值下计算累积分布函数（Cumulatuve Distribution Function, CDF）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，累积分布函数的值。

    .. py:method:: construct(name, *args, **kwargs)

        重写Cell中的 `construct` 。

        .. note:: 
            支持的函数包括：'prob'、'log_prob'、'cdf', 'log_cdf'、'survival_function'、'log_survival'、'var'、
            'sd'、'mode'、'mean'、'entropy'、'kl_loss'、'cross_entropy'、'sample'、'get_dist_args'、'get_dist_type'。

        参数：
            - **name** (str) - 函数名称。
            - **args** (list) - 函数所需的位置参数列表。
            - **kwargs** (dict) - 函数所需的关键字参数字典。

        返回：
            Tensor，name对应函数的值。

    .. py:method:: cross_entropy(dist, *args, **kwargs)

        计算分布a和b之间的交叉熵。

        参数：
            - **dist** (str) - 分布的类型。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            Distribution b的 `dist_spec_args` 必须通过 `args` 或 `kwargs` 传递给函数。传入Distribution a的 `dist_spec_args` 是可选的。

        返回：
            Tensor，交叉熵的值。

    .. py:method:: entropy(*args, **kwargs)

        计算熵。

        参数：
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，熵的值。

    .. py:method:: get_dist_args(*args, **kwargs)

        返回分布的参数列表。

        参数：
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note:: 
            `dist_spec_args` 必须以列表或者字典的形式传入。传递给字类的参数的顺序应该与通过 `_add_parameter` 初始化默认参数的顺序相同。如果某个 `dist_spec_args` 为None，那么将返回默认值。


        返回：
            list[Tensor]，参数列表。

    .. py:method:: get_dist_type()

        返回分布类型。

        返回：
            string，分布类型名字。

    .. py:method:: kl_loss(dist, *args, **kwargs)

        计算KL散度，即KL(a||b)。

        参数：
            - **dist** (str) - 分布的类型。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            Distribution b的 `dist_spec_args` 必须通过 `args` 或 `kwargs` 传递给函数。传入Distribution a的 `dist_spec_args` 是可选的。

        返回：
            Tensor，KL散度。

    .. py:method:: log_cdf(value, *args, **kwargs)

        计算给定值对于的累积分布函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_prob(value, *args, **kwargs)

        计算给定值对应的概率的对数（pdf或pmf）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，累积分布函数的对数。

    .. py:method:: log_survival(value, *args, **kwargs)

        计算给定值对应的生存函数的对数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，生存函数的对数。

    .. py:method:: mean(*args, **kwargs)

        计算期望。

        参数：
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，概率分布的期望。

    .. py:method:: mode(*args, **kwargs)

        计算众数。

        参数：
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，概率分布的众数。

    .. py:method:: prob(value, *args, **kwargs)

        计算给定值下的概率。对于离散分布是计算概率质量函数（Probability Mass Function），而对于连续分布是计算概率密度函数（Probability Density Function）。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，概率值。

    .. py:method:: sample(*args, **kwargs)

        采样函数。

        参数：
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，根据概率分布采样的样本。

    .. py:method:: sd(*args, **kwargs)

        计算标准差。

        参数：
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，概率分布的标准差。

    .. py:method:: survival_function(value, *args, **kwargs)

        计算给定值对应的生存函数。

        参数：
            - **value** (Tensor) - 要计算的值。
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，生存函数的值。

    .. py:method:: var(*args, **kwargs)

        计算方差。

        参数：
            - **args** (list) - 位置参数列表，具体需要的参数根据子类的实现确定。
            - **kwargs** (dict) - 关键字参数字典，具体需要的参数根据子类的实现确定。

        .. note::
            可以通过 `args` 或 `kwargs` 传递其 `dist_spec_args` 来选择性地将Distribution传递给函数。

        返回：
            Tensor，概率分布的方差。

