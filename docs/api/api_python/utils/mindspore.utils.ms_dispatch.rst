
.. py:class:: mindspore.utils.MsDispatchMode

    算子调度拦截的上下文管理器。

    进入上下文后，在该作用域内执行的所有算子都会先调用 ``__ms_dispatch__`` 回调方法。
    用户可以通过此机制自定义算子行为，例如日志记录、性能分析或修改输入/输出。

    MsDispatchMode 支持嵌套使用。嵌套时，最内层的模式优先处理。退出嵌套上下文后，
    外层模式恢复处理算子调度。

    使用时需继承该类并重写 ``__ms_dispatch__`` 方法。该类也可以作为装饰器使用。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    异常：
        - **TypeError** - 如果子类的 ``__ms_dispatch__`` 方法未正确接受参数 ``(func, args, kwargs)``。

    .. py:method:: __ms_dispatch__(func, args, kwargs=None)

        每个算子在上下文中执行时会调用的回调方法。在子类中重写此方法以自定义算子调度行为。

        参数：
            - **func** (OpFunc) - 原始算子的可调用包装器。

              - ``func.name``：算子名称（例如 "Add"、"MatMul"）。
              - ``func(*args)``：使用给定参数调用原始算子。

            - **args** (tuple) - 传递给算子的位置参数。
            - **kwargs** (dict, 可选) - 传递给算子的关键字参数。默认值： ``None``。

        返回：
            算子执行的结果。默认实现直接调用原始算子并原样返回其结果。

    样例：

    .. code-block:: python

        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>>
        >>> class OpLogger(ms.MsDispatchMode):
        ...     '''记录所有算子调用。'''
        ...     def __ms_dispatch__(self, func, args, kwargs=None):
        ...         print(f"Executing op: {func.name}")
        ...         result = func(*args)
        ...         print(f"Op {func.name} completed")
        ...         return result
        ...
        >>> x = Tensor([1.0, 2.0, 3.0])
        >>> y = Tensor([4.0, 5.0, 6.0])
        >>>
        >>> with OpLogger():
        ...     z = x + y
        Executing op: Add
        Op Add completed
        >>>
        >>> # 嵌套上下文
        >>> class InnerGuard(ms.MsDispatchMode):
        ...     def __ms_dispatch__(self, func, args, kwargs=None):
        ...         print(f"[Inner] {func.name}")
        ...         return func(*args)
        ...
        >>> with OpLogger():
        ...     a = x * 2
        ...     with InnerGuard():
        ...         b = x + y  # InnerGuard 优先处理
        ...     c = x - y  # OpLogger 恢复处理
        Executing op: Mul
        Op Mul completed
        [Inner] Add
        Executing op: Sub
        Op Sub completed
        >>>
        >>> # 装饰器用法
        >>> @OpLogger()
        ... def my_function(a, b):
        ...     return a + b
        ...
        >>> result = my_function(x, y)
        Executing op: Add
        Op Add completed
