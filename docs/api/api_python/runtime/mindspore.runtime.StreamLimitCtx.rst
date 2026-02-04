mindspore.runtime.StreamLimitCtx
================================

.. py:class:: mindspore.runtime.StreamLimitCtx(stream, cube_num=-1, vector_num=-1)

    上下文管理器，用于选择给定流的核数限制。

    上下文范围内的所有在给定流上执行的算子，都将指定cube和vector的核数。

    参数：
        - **stream** (:class:`mindspore.runtime.Stream`) - 指定的流。
        - **cube_num** (int，可选) - 设置流上的cube核数。默认值： ``-1``，表示不设置。
        - **vector_num** (int，可选) - 设置流上的vector核数。默认值： ``-1``，表示不设置。

    异常：
        - **TypeError** - 参数 `stream` 不是 :class:`mindspore.runtime.Stream` 。
