mindspore.ops.ScalarSummary
============================

.. py:class:: mindspore.ops.ScalarSummary

    通过标量汇总算子将标量输出到协议缓冲区。

    输入：
        - **name** (str) - 输入变量的名称，不能是空字符串。
        - **value** (Tensor) - 标量数据的值，维度必须为0或者1。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
