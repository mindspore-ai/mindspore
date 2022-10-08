mindspore.ops.ReduceOp
======================

.. py:class:: mindspore.ops.ReduceOp

    规约张量的操作选项。这是枚举类型，而不是运算符。

    主要调用方法如下：

    - SUM：ReduceOp.SUM.
    - MAX：ReduceOp.MAX.
    - MIN：ReduceOp.MIN.
    - PROD：ReduceOp.PROD.

    .. note::
        有关更多信息，请参阅示例。这需要在具有多个加速卡的环境中运行。
        在运行以下示例之前，用户需要预设环境变量。请参考官方网站 `MindSpore \
        <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html#通信算子>`_ 。

    有四种操作选项，"SUM"、"MAX"、"MIN"和"PROD"。

    - SUM：求和。
    - MAX：求最大值。
    - MIN：求最小值。
    - PROD：求乘积。
