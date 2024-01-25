mindspore.hal.StreamCtx
==========================

.. py:class:: mindspore.hal.StreamCtx(ctx_stream)

    上下文管理器，用于选择给定的流。

    上下文范围内的所有算子，都将在指定流上执行。

    参数：
        - **ctx_stream** (Stream) - 指定的流。如果是 `None` ，这个上下文管理器无操作。

    异常：
        - **TypeError** - 参数 `ctx_stream` 即不是一个 :class:`mindspore.hal.Stream` 也不是一个 ``None``。
