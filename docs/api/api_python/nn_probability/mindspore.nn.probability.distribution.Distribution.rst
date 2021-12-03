mindspore.nn.probability.distribution.Distribution
===================================================

.. py:class:: mindspore.nn.probability.distribution.Distribution(seed, dtype, name, param)

    所有分布的基类。

    **参数：**

    - **seed** (int) - 采样时使用的种子。如果为None，则使用0。
    - **dtype** (mindspore.dtype) - 事件样例的类型。
    - **name** (str) - 分布的名称。
    - **param** (dict) - 用于初始化分布的参数。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        派生类必须重写 `_mean` 、 `_prob` 和 `_log_prob` 等操作。必填参数必须通过 `args` 或 `kwargs` 传入，如 `_prob` 的 `value` 。
    
    .. py:method:: cdf(value, *args, **kwargs)

        在给定值下评估累积分布函数（Cumulatuve Distribution Function, CDF）。

        **参数：**

        - **value** (Tensor) - 要评估的值。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: construct(name, *args, **kwargs)

        重写Cell中的 `construct` 。

        .. note:: 
            支持的函数包括：'prob'、'log_prob'、'cdf', 'log_cdf'、'survival_function'、'log_survival'、'var'、
            'sd'、'mode'、'mean'、'entropy'、'kl_loss'、'cross_entropy'、'sample'、'get_dist_args'、'get_dist_type'。

        **参数：**

        - **name** (str) - 函数名称。
        - **args** (list) - 函数所需的位置参数列表。
        - **kwargs** (dict) - 函数所需的关键字参数字典。
        
    .. py:method:: cross_entropy(dist, *args, **kwargs)

        评估分布a和b之间的交叉熵。

        **参数：**

        - **dist** (str) - 分布的类型。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: entropy(*args, **kwargs)

        计算熵。

        **参数：**

        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: get_dist_args(*args, **kwargs)

        检查默认参数的可用性和有效性。

        **参数：**

        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。

        .. note:: 
           传递给字类的参数的顺序应该与通过 `_add_parameter` 初始化默认参数的顺序相同。
        
    .. py:method:: get_dist_type()

        返回分布类型。
        
    .. py:method:: kl_loss(dist, *args, **kwargs)

        评估KL散度，即KL(a||b)。

        **参数：**

        - **dist** (str) - 分布的类型。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: log_cdf(value, *args, **kwargs)

        计算给定值对于的cdf的对数。

        **参数：**

        - **value** (Tensor) - 要评估的值。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: log_prob(value, *args, **kwargs)

        计算给定值对应的概率的对数（pdf或pmf）。

        **参数：**

        - **value** (Tensor) - 要评估的值。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: log_survival(value, *args, **kwargs)

        计算给定值对应的剩余函数的对数。

        **参数：**

        - **value** (Tensor) - 要评估的值。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: mean(*args, **kwargs)

        评估平均值。

        **参数：**

        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: mode(*args, **kwargs)

        评估模式。

        **参数：**

        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: prob(value, *args, **kwargs)

        评估给定值下的概率（Probability Density Function或Probability Mass Function）。

        **参数：**

        - **value** (Tensor) - 要评估的值。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: sample(*args, **kwargs)

        采样函数。

        **参数：**

        - **shape** (tuple) - 样本的shape。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: sd(*args, **kwargs)

        标准差评估。

        **参数：**

        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: survival_function(value, *args, **kwargs)

        计算给定值对应的剩余函数。

        **参数：**

        - **value** (Tensor) - 要评估的值。
        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
    .. py:method:: var(*args, **kwargs)

        评估方差。

        **参数：**

        - **args** (list) - 传递给子类的位置参数列表。
        - **kwargs** (dict) - 传递给子类的关键字参数字典。
        
