mindspore.ops.prim_attr_register
================================

.. py:function:: mindspore.ops.prim_attr_register(fn)

    Primitive属性的注册器。

    注册装饰器，其中装饰器用于内置算子的Primitive的'__init__'函数。该函数将添加'__init__'的所有参数作为算子属性，并且初始化Primitive的名称。

    参数：
        - **fn** (function) - Primitive的__init__函数。

    返回：
        函数，原始函数。
