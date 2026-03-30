mindspore.ops.unstack
=======================

.. py:function:: mindspore.ops.unstack(input_x, axis=0)

    按指定轴对输入tensor进行拆分。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **axis** (int) - 指定轴。默认 ``0`` 。

    返回：
        多个tensor组成的tuple。
