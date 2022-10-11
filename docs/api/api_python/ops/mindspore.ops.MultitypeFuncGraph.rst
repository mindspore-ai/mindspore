mindspore.ops.MultitypeFuncGraph
================================

.. py:class:: mindspore.ops.MultitypeFuncGraph(name, read_value=False)

    MultitypeFuncGraph是一个用于生成重载函数的类，使用不同类型作为输入。使用 `name` 去初始化一个MultitypeFuncGraph，并且使用带有
    类型的 `register` 注册器进行装饰注册类型。这样使该函数可以使用不同的类型作为输入调用，一般与 `HyperMap` 与 `Map` 结合使用。

    参数：
        - **name** (str) - 操作名。
        - **read_value** (bool, 可选) - 如果注册函数不需要对输入的值进行更改，即所有输入都为按值传递，则将 `read_value` 设置为True。默认为：False。

    异常：
        - **ValueError** - 找不到给定参数类型所匹配的函数。

    .. py:method:: register(*type_names)

        根据给出的字符串内容注册不同输入类型的函数。

        参数：
             - **type_names** (Union[str, :class:`mindspore.dtype`]) - 输入类型的名或者一个类型列表。

        返回：
            装饰器， 一个根据 `type_names` 指定输入类型的注册函数的装饰器。

