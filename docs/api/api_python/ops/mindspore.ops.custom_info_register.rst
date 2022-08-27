mindspore.ops.custom_info_register
==================================

.. py:class:: mindspore.ops.custom_info_register(*reg_info)

    装饰器，用于将注册信息绑定到： :class:`mindspore.ops.Custom` 的 `func` 参数。

    .. note::
        `reg_info` 将添加到算子库'oplib'中。
        
    参数：
        - **reg_info** (tuple[str, dict]) - json格式的算子注册信息。
        
    返回：
        function，返回算子信息注册的装饰器。
