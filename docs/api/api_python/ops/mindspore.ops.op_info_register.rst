mindspore.ops.op_info_register
===============================

.. py:function:: mindspore.ops.op_info_register(op_info)

    用于注册算子的装饰器。

    .. note::
        'op_info'应通过json格式的字符串表示算子信息。'op_info'将添加到算子库'oplib'中。

    参数：
        - **op_info** (Union[str, dict]) - json格式的算子信息。

    返回：
        Function，返回算子信息注册的装饰器。
    