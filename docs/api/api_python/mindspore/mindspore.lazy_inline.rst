mindspore.lazy_inline
=====================

.. py:function:: mindspore.lazy_inline(fn=None, attrs=None, policy=None)

    指定一个cell是可复用的。该cell在前端编译为可复用的子图，后端根据策略内联。
    注册此装饰器到cell的内置函数 `__init__` 时，此装饰器会按照 `attrs` 的值去添加 `__init__` 函数对应的入参，以此作为cell的属性。

    详细功能说明可参考 `使用lazy_inline装饰器 <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph_expert_programming.html#%E4%BD%BF%E7%94%A8lazy-inline%E8%A3%85%E9%A5%B0%E5%99%A8>`_ 。

    .. warning::
        - 该特性仅支持Ascend，其他硬件不支持。
        - cell的construct函数参数必须是位置参数或者关键字参数，且不能有默认值。
        - lazy inline 装饰的cell不包含控制流。
        - 梯度累加场景下，推荐使用@lazy_inline装饰器来缩短编译时间，并且仅支持将@lazy_inline装饰器配置在最外层的Cell上。

    参数：
        - **fn** (function，可选) - cell的 `__init__` 函数。默认值： ``None``。
        - **attrs** (Union[list[string], string]，可选) - cell需要添加的属性列表。默认值： ``None``。
        - **policy** (Union[None, "front"]，可选) - inline 的策略。默认值： ``None``。

          - ``None``: Cell编译为可复用的子图，该子图不inline到大图中。
          - ``"front"``: Cell先编译为可复用的子图，然后inline到大图中。

    返回：
        function，原始函数。
