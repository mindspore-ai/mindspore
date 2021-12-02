mindspore.nn.probability.bijector.Bijector
===========================================

.. py:class:: mindspore.nn.probability.bijector.Bijector(is_constant_jacobian=False, is_injective=True, name=None, dtype=None, param=None)

    Bijector类。

    **参数：**
    
    - **is_constant_jacobian** (bool) - Bijector是否具有常数导数。默认值：False。
    - **is_injective** (bool) - Bijector是否为一对一映射。默认值：True。
    - **name** (str) - Bijector名称。默认值：None。
    - **dtype** (mindspore.dtype) - Bijector可以操作的分布的数据类型。默认值：None。
    - **param** (dict) - 用于初始化Bijector的参数。默认值：None。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note::
        Bijector的 `dtype` 为None时，输入值必须是float类型，除此之外没有其他强制要求。在初始化过程中，当 `dtype` 为None时，对参数的数据类型没有强制要求。但所有参数都应具有相同的float类型，否则将引发TypeError。具体来说，参数类型跟随输入值的数据类型，即当 `dtype` 为None时，Bijector的参数将被强制转换为与输入值相同的类型。当指定了 `dtype` 时，参数和输入值的 `dtype` 必须相同。当参数类型或输入值类型与 `dtype` 不相同时，将引发TypeError。只能使用mindspore的float数据类型来指定Bijector的 `dtype` 。
    
    .. py:method:: cast_param_by_value(value, para)

        将Bijector的参数para的数据类型转换为与value相同的类型。
        
        **参数：**

        - **value** (Tensor) - 输入数据。
        - **para** (Tensor) - Bijector参数。
        
    .. py:method:: construct(name, *args, **kwargs)

        重写Cell中的 `construct` 。

    .. note::
        支持的函数包括：'forward'、'inverse'、'forward_log_jacobian'、'inverse_log_jacobian'。

        **参数：**
        
        - **name** (str) - 函数名称。
        - **args** (list) - 函数所需的位置参数列表。
        - **kwargs** (dict) - 函数所需的关键字参数字典。
        
    .. py:method:: forward(value, *args, **kwargs)

        正变换：将输入值转换为另一个分布。
        
        **参数：**

        - **value** (Tensor) - 输入。
        - **args** (list) - 函数所需的位置参数列表。
        - **kwargs** (dict) - 函数所需的关键字参数字典。
        
    .. py:method:: forward_log_jacobian(value, *args, **kwargs)

        对正变换导数取对数。
        
        **参数：**

        - **value** (Tensor) - 输入。
        - **args** (list) - 函数所需的位置参数列表。
        - **kwargs** (dict) - 函数所需的关键字参数字典。
        
    .. py:method:: inverse(value, *args, **kwargs)

        逆变换：将输入值转换回原始分布。
        
        **参数：**
        
        - **value** (Tensor) - 输入。
        - **args** (list) - 函数所需的位置参数列表。
        - **kwargs** (dict) - 函数所需的关键字参数字典。
        
    .. py:method:: inverse_log_jacobian(value, *args, **kwargs)

        对逆变换的导数取对数。
        
        **参数：**

        - **value** (Tensor) - 输入。
        - **args** (list) - 函数所需的位置参数列表。
        - **kwargs** (dict) - 函数所需的关键字参数字典。
        