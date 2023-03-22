mindspore.nn.probability.bijector.Bijector
===========================================

.. py:class:: mindspore.nn.probability.bijector.Bijector(is_constant_jacobian=False, is_injective=True, name=None, dtype=None, param=None)

    Bijector类。Bijector描述了一种随机变量的映射方法。可以通过一个已有的随机变量 :math:`X` 和一个映射函数 :math:`g(x)` 生成一个新的随机变量 :math:`Y = g(X)` 。

    参数：    
        - **is_constant_jacobian** (bool) - Bijector是否具有常数导数。默认值：False。
        - **is_injective** (bool) - Bijector是否为一对一映射。默认值：True。
        - **name** (str) - Bijector名称。默认值：None。
        - **dtype** (mindspore.dtype) - Bijector可以操作的分布的数据类型。默认值：None。
        - **param** (dict) - 用于初始化Bijector的参数。默认值：None。

    .. note::
        Bijector的 `dtype` 为None时，输入值必须是float类型，除此之外没有其他强制要求。

        在初始化过程中，当 `dtype` 为None时，对参数的数据类型没有强制要求。

        但所有参数都应具有相同的float类型，否则将引发TypeError。

        具体来说，参数类型跟随输入值的数据类型。即当 `dtype` 为None时，Bijector的参数将被强制转换为与输入值相同的类型。

        当指定了 `dtype` 时，参数和输入值的 `dtype` 必须相同。

        当参数类型或输入值类型与 `dtype` 不相同时，将引发TypeError。只能使用mindspore.float_type数据类型来指定Bijector的 `dtype` 。

    .. py:method:: cast_param_by_value(value, para)

        将输入中的 `para` 的数据类型转换为与 `value` 相同的类型，一般由Bijector的子类用于基于输入对自身参数进行数据类型变化。

        参数：
            - **value** (Tensor) - 输入数据。
            - **para** (Tensor) - Bijector参数。

        返回：
            Tensor，参数经过数据类型转换之后的值。
        
    .. py:method:: construct(name, *args, **kwargs)

        重写Cell中的 `construct` 。

        .. note::
            支持的函数名称包括：'forward'、'inverse'、'forward_log_jacobian'、'inverse_log_jacobian'。

        参数：        
            - **name** (str) - 函数名称。
            - **args** (list) - 函数所需的位置参数列表。
            - **kwargs** (dict) - 函数所需的关键字参数字典。
        
        返回：
            Tensor，name对应函数的值。

    .. py:method:: forward(value, *args, **kwargs)

        正映射，计算输入随机变量经过映射后的值。
        
        参数：
            - **value** (Tensor) - 输入随机变量的值。
            - **args** (list) - 函数所需的位置参数列表。
            - **kwargs** (dict) - 函数所需的关键字参数字典。

        返回：
            Tensor，输出随机变量的值。
        
    .. py:method:: forward_log_jacobian(value, *args, **kwargs)

        计算正映射导数的对数值。
        
        参数：
            - **value** (Tensor) - 输入随机变量的值。
            - **args** (list) - 函数所需的位置参数列表。
            - **kwargs** (dict) - 函数所需的关键字参数字典。

        返回：
            Tensor，正映射导数的对数值。
        
    .. py:method:: inverse(value, *args, **kwargs)

        逆映射，计算输出随机变量对应的输入随机变量的值。
        
        参数：        
            - **value** (Tensor) - 输出随机变量的值。
            - **args** (list) - 函数所需的位置参数列表。
            - **kwargs** (dict) - 函数所需的关键字参数字典。

        返回：
            Tensor，输入随机变量的值。

    .. py:method:: inverse_log_jacobian(value, *args, **kwargs)

        计算逆映射导数的对数值。

        参数：
            - **value** (Tensor) - 输出随机变量的值。
            - **args** (list) - 函数所需的位置参数列表。
            - **kwargs** (dict) - 函数所需的关键字参数字典。

        返回：
            Tensor，逆映射导数的对数值。
