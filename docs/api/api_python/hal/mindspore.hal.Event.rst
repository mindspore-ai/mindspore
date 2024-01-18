mindspore.hal.Event
========================

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

        如果未指定 `stream`，将使用 :func:`mindspore.hal.current_stream` 。

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
