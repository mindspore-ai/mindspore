mindspore.ops.IsInstance
=========================

.. py:class:: mindspore.ops.IsInstance

    检查输入对象是否为目标类型的实例。

    **输入：**

    - **inst** (Any Object) - 要检查的实例。只允许为常量。
    - **type_** (mindspore.dtype) - 目标类型。只允许为常量。

    **输出：**

    bool，检查结果。

    **异常：**

    - **TypeError** - 如果 `type_` 不是一种类型。