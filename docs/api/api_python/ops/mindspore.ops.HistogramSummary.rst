mindspore.ops.HistogramSummary
===============================

.. py:class:: mindspore.ops.HistogramSummary

    将Tensor数据转换为直方图数据记录。

    输入：
        - **name** (str) - 输入变量的名称。
        - **value** (Tensor) - Tensor的值，Tensor的rank必须大于0。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
