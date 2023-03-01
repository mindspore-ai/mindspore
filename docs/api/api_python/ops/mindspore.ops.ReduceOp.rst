mindspore.ops.ReduceOp
======================

.. py:class:: mindspore.ops.ReduceOp

    规约张量的操作选项。这是枚举类型，而不是运算符。

    主要调用方法如下：

    - SUM：ReduceOp.SUM.
    - MAX：ReduceOp.MAX.
    - MIN：ReduceOp.MIN.
    - PROD：ReduceOp.PROD.

    有四种操作选项，"SUM"、"MAX"、"MIN"和"PROD"。

    - SUM：求和。
    - MAX：求最大值。
    - MIN：求最小值。
    - PROD：求乘积。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
