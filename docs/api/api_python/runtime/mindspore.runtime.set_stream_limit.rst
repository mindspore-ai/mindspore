mindspore.runtime.set_stream_limit
==================================

.. py:function:: mindspore.runtime.set_stream_limit(stream, cube_num=-1, vector_num=-1)

    设置指定流上的限制核数。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **stream** (:class:`mindspore.runtime.Stream`) - 指定的流对象。
        - **cube_num** (int，可选) - 设置流上的cube核数。默认值： ``-1``，表示不设置。
        - **vector_num** (int，可选) - 设置流上的vector核数。默认值： ``-1``，表示不设置。
