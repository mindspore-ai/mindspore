mindspore.frombuffer
=====================

.. py:function:: mindspore.frombuffer(buffer, dtype, count=-1, offset=0)

    从实现了缓冲区接口的对象创建一维tensor。返回的tensor与缓冲区共享同一内存空间。

    参数：
        - **buffer** (buffer) - 实现了缓冲区接口的对象。
        - **dtype** (mindspore.dtype) - 返回tensor的期望数据类型。
        - **count** (int, 可选) - 要读取的数据项数量。默认值： ``-1`` 。
        - **offset** (int, 可选) - 起始位置的字节偏移量。默认值： ``0`` 。

    返回：
        Tensor，包含缓冲区数据的一维Tensor。

    异常：
        - **RuntimeError** - 如果 `count` 为小于-1的负数。
        - **RuntimeError** - 如果缓冲区长度为小于等于0或者 `count` 为0。
        - **RuntimeError** - 如果 `offset` 不在[0, 缓冲区长度-1]范围内。
        - **RuntimeError** - 当 `count` 为-1时，如果缓冲区长度减去 `offset` 后不能整除元素大小。
        - **RuntimeError** - 如果 `offset` 后的剩余缓冲区大小不足以容纳请求的元素数量。
