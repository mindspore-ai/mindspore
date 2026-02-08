mindspore.set_recursion_limit
=============================

.. py:function:: mindspore.set_recursion_limit(recursion_limit=1000)

    在图编译前指定函数调用的递归深度限制。
    当嵌套的函数调用过深或者子图数量过多时，需要调用该接口。如果 `recursion_limit` 被设置成大于以前的值，那么系统最大栈深度也要被设置成更大，否则会因为系统栈溢出而引起一个 `core dumped` 异常。

    参数：
        - **recursion_limit** (int, 可选) - 递归深度限制。必须为正整数。默认值： ``1000`` 。
