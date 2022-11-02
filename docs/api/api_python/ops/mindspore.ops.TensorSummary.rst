mindspore.ops.TensorSummary
============================

.. py:class:: mindspore.ops.TensorSummary

    通过TensorSummary将一个Tensor输出到协议缓冲区。

    输入：
        - **name** (str) - 输入变量的名称。
        - **value** (Tensor) - Tensor的值，Tensor的维度必须大于0。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
        - **ValueError** - 如果 `value` 的维度等于0。

