mindspore.nn.rearrange_inputs
==============================

.. py:function:: mindspore.nn.rearrange_inputs(func)

    此装饰器用于根据类的 `indexes` 属性对输入重新排列。

    此装饰器目前用于 :class:`mindspore.nn.Metric` 类的 `update` 方法。

    **样例：**

    >>> class RearrangeInputsExample:
    ...     def __init__(self):
    ...         self._indexes = None
    ...
    ...     @property
    ...     def indexes(self):
    ...         return getattr(self, '_indexes', None)
    ...
    ...     def set_indexes(self, indexes):
    ...         self._indexes = indexes
    ...         return self
    ...
    ...     @rearrange_inputs
    ...     def update(self, *inputs):
    ...         return inputs
    >>>
    >>> rearrange_inputs_example = RearrangeInputsExample().set_indexes([1, 0])
    >>> outs = rearrange_inputs_example.update(5, 9)
    >>> print(outs)
    (9, 5)

    **参数：**

    - **func** (Callable) - 要装饰的候选函数，其输入将被重新排列。

    **返回：**

    Callable，用于在函数之间调换输入。
