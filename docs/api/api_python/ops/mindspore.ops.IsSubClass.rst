mindspore.ops.IsSubClass
=========================

.. py:class:: mindspore.ops.IsSubClass

    检查输入类型是否为其他类型的子类。

    **输入：**

    - **sub_type** (mindspore.dtype) - 要检查的类型。只允许为常量。
    - **type_** (mindspore.dtype) - 目标类型。只允许为常量。

    **输出：**

    bool，检查结果。

    **异常：**

    - **TypeError** - 如果 `sub_type` 或 `type_` 不是一种类型。