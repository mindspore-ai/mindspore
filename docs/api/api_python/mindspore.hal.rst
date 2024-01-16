mindspore.hal
=============

Hal中封装了设备管理、流管理与事件管理的接口。MindSpore从不同后端抽象出对应的上述模块，允许用户在Python层调度硬件资源，并且这些接口目前只在PyNative模式下生效。

.. py:class:: mindspore.hal.Event(enable_timing=False, blocking=False)

    设备事件的封装器。

    设备事件是同步标记，可用于监视设备的执行进度、准确计时和同步设备流。

    当事件首次被记录时，底层设备事件才会被初始化。

    参数：
        - **enable_timing** (bool, 可选) - 事件是否应计时。（默认值： ``False``）
        - **blocking** (bool, 可选) - 如果为 ``True``， `wait` 函数将是阻塞的。（默认值： ``False``）

    .. py:method:: mindspore.hal.Event.elapsed_time(end_event)

        返回记录事件之后到记录end_event之前所用的时间（以毫秒为单位）。

        参数：
            - **end_event** (Event) - 结束事件。

        返回：
            float，经过的时间（以毫秒为单位）。

        异常：
            - **TypeError** - 参数 `end_event` 不是一个 :class:`mindspore.hal.Event`。

    .. py:method:: mindspore.hal.Event.query()

        检查事件当前捕获的所有工作是否已完成。

        返回：
            bool，指示事件当前捕获的所有工作是否都已完成。

    .. py:method:: mindspore.hal.Event.record(stream=None)

        在给定的流中记录事件。

        如果未指定 `stream`，将使用 :func:`mindspore.hal.current_stream()` 。

        参数：
            - **stream** (Stream, 可选) - 需要记录的流。如果输入为 ``None``，将使用当前流。默认值： ``None``。

        异常：
            - **TypeError** - 参数 `stream` 即不是一个 :class:`mindspore.hal.Stream` 也不是一个 ``None``。

    .. py:method:: mindspore.hal.Event.synchronize()

        等待事件完成。

        等待直到完成当前此事件捕获的所有工作。
        这将阻止CPU线程继续进行，直到事件完成。

    .. py:method:: mindspore.hal.Event.wait(stream=None)

        使提交给给定流的所有未来工作等待此事件。

        如果未指定 `stream`，将使用 :func:`mindspore.hal.current_stream()` 。

        参数：
            - **stream** (Stream, 可选) - 需要等待的流。如果输入为 ``None``，将使用当前流。默认值： ``None``。

        异常：
            - **TypeError** - 参数 `stream` 即不是一个 :class:`mindspore.hal.Stream` 也不是一个 ``None``。

.. py:class:: mindspore.hal.Stream(priority=0, **kwargs)

    基于设备流的封装器。
    设备流是属于特定设备的线性执行序列，流之间相互独立。

    参数：
        - **priority** (int, 可选) - 流的优先级，较低的数字表示较高的优先级。默认情况下，流的优先级为 ``0``。
        - **kwargs** (dict) - 关键字参数字典。

    .. py:method:: mindspore.hal.Stream.query()

        检查所有提交的工作是否已完成。

        返回：
            bool，指示该流中的所有算子是否已执行完成。

    .. py:method:: mindspore.hal.Stream.record_event(event=None)

        记录一个事件。

        参数：
            - **event** (Event, 可选) - 要记录的事件。如果输入为 ``None``，将分配一个新的事件。默认值： ``None``。

        返回：
            Event，记录的事件。

        异常：
            - **TypeError** - 参数 `event` 即不是一个 :class:`mindspore.hal.Event` 也不是一个 ``None``。

    .. py:method:: mindspore.hal.Stream.synchronize()

        等待此流中的所有算子执行完成。

    .. py:method:: mindspore.hal.Stream.wait_event(event)

        使提交到流的所有未来工作等待本事件。

        参数：
            - **event** (Event) - 等待的事件。

        异常：
            - **TypeError** - 参数 `event` 不是一个 :class:`mindspore.hal.Event`。

    .. py:method:: mindspore.hal.Stream.wait_stream(stream)

        与另一个流同步。

        所有提交到本流的未来工作都将等待，直到所有算子都提交至给定流并执行完成。

        参数：
            - **stream** (Stream) - 需要同步的流。

        异常：
            - **TypeError** - 参数 `stream` 不是一个 :class:`mindspore.hal.Stream` 。

.. py:class:: mindspore.hal.StreamCtx(ctx_stream)

    上下文管理器，用于选择给定的流。

    上下文范围内的所有算子，都将在指定流上执行。

    参数：
        - **ctx_stream** (Stream) - 指定的流。如果是 `None` ，这个上下文管理器无操作。

    异常：
        - **TypeError** - 参数 `ctx_stream` 即不是一个 :class:`mindspore.hal.Stream` 也不是一个 ``None``。

.. py:function:: mindspore.hal.current_stream()

    返回此设备上正在使用的流。

    返回：
        Stream，此设备上正在使用的流。

.. py:function:: mindspore.hal.default_stream()

    返回此设备上的默认流。

    返回：
        Stream，此设备上的默认流。

.. py:function:: mindspore.hal.set_cur_stream(stream)

    设置当前流，这是用于设置流的包装器API。

    不建议使用此函数，建议使用 `StreamCtx` 上下文管理器。

    参数：
        - **stream** (Stream) - 指定的流。如果是 ``None`` ，这个上下文管理器无操作。

    异常：
        - **TypeError** - 参数 `stream` 即不是一个 :class:`mindspore.hal.Stream` 也不是一个 ``None``。

.. py:function:: mindspore.hal.synchronize()

    同步当前设备上的所有流。（每个MindSpore进程只占用一个设备）
