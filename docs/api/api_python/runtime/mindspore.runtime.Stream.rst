mindspore.runtime.Stream
===========================

.. py:class:: mindspore.runtime.Stream(priority=0, **kwargs)

    基于设备流的封装器。
    设备流是属于特定设备的线性执行序列，流之间相互独立。

    参数：
        - **priority** (int, 可选) - 流的优先级，较低的数字表示较高的优先级。默认 ``0``。
        - **kwargs** (dict) - 关键字参数字典。

    .. py:method:: mindspore.runtime.Stream.query()

        检查所有提交的工作是否已完成。

        返回：
            bool，指示该流中的所有算子是否已执行完成。

    .. py:method:: mindspore.runtime.Stream.record_event(event=None)

        记录一个事件。

        参数：
            - **event** (Event, 可选) - 要记录的事件。如果输入为 ``None``，将分配一个新的事件。默认 ``None``。

        返回：
            Event，记录的事件。

        异常：
            - **TypeError** - 参数 `event` 既不是一个 :class:`mindspore.runtime.Event` 也不是一个 ``None``。

    .. py:method:: mindspore.runtime.Stream.synchronize()

        等待此流中的所有算子执行完成。

    .. py:method:: mindspore.runtime.Stream.wait_event(event)

        使提交到流的所有未来工作等待本事件。

        参数：
            - **event** (Event) - 等待的事件。

        异常：
            - **TypeError** - 参数 `event` 不是一个 :class:`mindspore.runtime.Event`。

    .. py:method:: mindspore.runtime.Stream.wait_stream(stream)

        与另一个流同步。

        所有提交到本流的未来工作都将等待，直到所有算子都提交至给定流并执行完成。

        参数：
            - **stream** (Stream) - 需要同步的流。

        异常：
            - **TypeError** - 参数 `stream` 不是一个 :class:`mindspore.runtime.Stream` 。
