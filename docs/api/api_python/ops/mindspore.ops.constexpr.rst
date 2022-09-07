mindspore.ops.constexpr
=======================

.. py:function:: mindspore.ops.constexpr(fn=None, get_instance=True, name=None, reuse_result=True, check=True)

    创建PrimiveWithInfer算子，用于在编译时推断值。可以用它定义函数，从而使用构造函数中的常量计算出常量值。

    参数：
        - **fn** (function) - `fn` 用作输出算子的infer_value。默认值：None。
        - **get_instance** (bool) - 如果为True，返回算子的实例，否则返回算子的类。默认值：True。
        - **name** (str) - 定义算子的名称。如果 `name` 为None，则使用函数名称作为算子名称。默认值：None。
        - **reuse_result** (bool) - 如果为True，仅实际执行算子一次，后续调用会直接返回结果。否则每次都实际执行算子来获取结果。默认值：True。
        - **check** (bool) - 如果为True，参数将被check，如果参数不是常量值，将发出警告信息。默认值：True。