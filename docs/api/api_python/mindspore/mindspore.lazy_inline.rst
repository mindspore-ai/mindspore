mindspore.lazy_inline
=====================

.. py:function:: mindspore.lazy_inline(fn=None, attrs=None)

    指定一个cell是可复用的。该cell在前端编译为可复用的子图，后端根据策略内联。
    注册此装饰器到cell的内置函数 `__init__` 时，此装饰器会按照 `attrs` 的值去添加 `__init__` 函数对应的入参作为cell的属性。

    .. warning::
        该特性仅支持Ascend，其它硬件不支持。

    参数：
        - **fn** (function) - cell的 `__init__` 函数。
        - **attrs** (Union[list[string], string]) - cell需要添加的属性列表。

    返回：
        function，原始函数。
