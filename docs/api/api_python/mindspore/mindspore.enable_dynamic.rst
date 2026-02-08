mindspore.enable_dynamic
========================

.. py:function:: mindspore.enable_dynamic(**kwargs)

    用于指定参数的shape为动态shape或动态rank。

    .. note::
        - 需要配合jit接口使用，不使用jit装饰器则无法使能动态shape和动态rank功能。
        - 在 `set_context(mode=GRAPH_MODE)` 和 `nn.Cell` 同时使用的场景下，使用 `enable_dynamic` 将会报错。

    参数：
        - **\*\*kwargs** (dict[str, Union[Tensor, tuple[Tensor], list[Tensor]]]) - 参数名称到动态shape配置的映射字典。字典的键为函数参数名，字典的值可以为Tensor、tuple[Tensor]或list[Tensor]类型。如需指定参数的shape中一个或多个维度为动态shape，可将shape中相应维度设置为None。如需指定参数的shape为动态rank，可将shape设置为None。

    返回：
        Function，装饰器函数，用于为被装饰的函数指定参数的动态shape信息。
