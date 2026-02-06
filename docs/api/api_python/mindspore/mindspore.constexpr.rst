mindspore.constexpr
=======================

.. py:function:: mindspore.constexpr(fn=None, get_instance=True, name=None, reuse_result=True, check=True)

    在图模式下，用来计算图编译过程中的常量值，以提升编译性能。

    参数：
        - **fn** (function，可选) - 用作输出算子的infer_value的函数。默认值： ``None`` 。
        - **get_instance** (bool，可选) - 如果为 ``True`` ，返回算子的实例，否则返回算子的类。默认值： ``True`` 。
        - **name** (str，可选) - 定义算子的名称。如果 `name` 为 ``None``，则使用函数名称作为算子名称。默认值： ``None`` 。
        - **reuse_result** (bool，可选) - 如果为 ``True`` ，仅实际执行算子一次，后续调用会直接返回结果。否则每次都实际执行算子来获取结果。默认值： ``True`` 。
        - **check** (bool，可选) - 如果为 ``True`` ，参数将被检查。如果参数不是常量，将发出警告信息。默认值： ``True`` 。