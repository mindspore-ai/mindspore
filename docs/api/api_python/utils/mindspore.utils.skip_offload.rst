
.. py:function:: mindspore.utils.skip_offload(func)

    跳过函数前向和反向过程卸载的装饰器。

    该装饰器用于包装一个函数或方法。调用时，它会在该函数的前向执行期间临时关闭卸载；同时会在该函数的输入和输出张量上注册反向hook，
    以便在该函数的反向执行期间临时关闭追踪。

    主要用于 :func:`mindspore.nn.Cell.auto_offload` 开启自动卸载后，对包含动态控制流或无法被追踪的代码段跳过卸载。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **func** (function) - 被包装的函数。

    返回：
        function，包装后的函数。
