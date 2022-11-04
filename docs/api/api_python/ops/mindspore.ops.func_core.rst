mindspore.ops.core
===================

.. py:function:: mindspore.ops.core(fn=None, **flags)

    向函数添加标志的装饰器。
    默认情况下，函数标记为True，允许使用此装饰器为图添加标志。

    参数：
        - **fn** (Function，可选) - 添加标志的函数。默认值：None。
        - **flags** (dict，可选) - 以下标志可以设置为core，表示这是核心函数或其他函数。默认值：None。
