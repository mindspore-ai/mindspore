
.. py:class:: mindspore.utils.MsDispatchMode

    算子调度拦截的上下文管理器。

    进入上下文后，在该作用域内执行的所有算子都会先调用 ``__ms_dispatch__`` 回调方法。
    用户可以通过此机制自定义算子行为，例如日志记录、性能分析或修改输入/输出。

    MsDispatchMode 支持嵌套使用。嵌套时，最内层的模式优先处理。退出嵌套上下文后，
    外层模式恢复处理算子调度。

    使用时需继承该类并重写 ``__ms_dispatch__`` 方法。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        - 该上下文管理器也可以作为装饰器使用。
        - ``__ms_dispatch__`` 方法接收一个包装了原始算子调用的 ``OpFunc`` 对象以及算子的参数。

    异常：
        - **TypeError** - 如果子类的 ``__ms_dispatch__`` 方法未正确接受参数 ``(func, args, kwargs)``。
