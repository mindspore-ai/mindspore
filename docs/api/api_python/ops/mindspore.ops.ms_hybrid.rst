mindspore.ops.ms_hybrid
=======================

.. py:function:: mindspore.ops.ms_hybrid(fn=None, reg_info=None, compile_attrs=None)

    用于MindSpore Hybrid DSL函数书写的装饰器。
    给用MindSpore Hybrid DSL书写的函数加上此装饰器后，它可以用作一个普通的Python函数。
    与此同时，他可以用于自定义算子:class:`mindspore.ops.Custom`的输入，其对应的`func_type`可以设置为`hybrid`或者`pyfunc`。
    使用`hybrid`类型的:class:`mindspore.ops.Custom`自定义算子可以自动推导数据类型和形状。

    **参数：**

    - **fn** (Function) - MindSpore Hybrid DSL书写的函数。默认值：None。
    - **reg_info** (tuple[str, dict]) - 算子注册信息。默认值：None。
    - **compile_attrs** (Dict) - 算子编译信息。默认值：None。

    **输出：**

    Function，如果`fn`不是None，那么返回一个用Hybrid DSL写的可执行函数；如果`fn`是None，则返回一个装饰器，该装饰器只有`fn`一个参数。