mindspore.lazy_inline
=====================

.. py:function:: mindspore.lazy_inline(fn=None, attrs=None)

    指定一个cell是可复用的。该cell对应的子图在编译时会被延迟inline。

    注册此装饰器到cell的内置函数 `__init__` 时，此装饰器会按照 `attrs` 的值去添加 `__init__` 函数对应的入参作为cell的属性。


    参数：
        - **fn** (function) - cell的 `__init__` 函数。
        - **attrs** (Union[list[string], string]) - cell需要添加的属性列表。

    返回：
        function，原始函数。
