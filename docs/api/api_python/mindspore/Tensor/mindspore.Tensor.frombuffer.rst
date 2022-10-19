mindspore.Tensor.frombuffer
============================

.. py:method:: mindspore.Tensor.frombuffer(buffer, dtype=mstype.float64, count=-1, offset=0)

    从实现Python缓冲区协议的对象创建一维`Tensor`。跳过缓冲区中的`offset`字节，并且取出数据类型为`dtype`的`count`个数据。

    参数：
        buffer（object）：公开缓冲区接口的Python对象。
        dtype（mindspore.dtype）：返回张量的所需数据类型。
        count（int，可选）：要读取的所需元素的数量。如果为负值，将读取所有元素（直到缓冲区结束）。默认值：-1。
        offset（int，可选）：缓冲区开始时要跳过的字节数。默认值：0

    返回：
        来自实现Python缓冲协议的对象的一维张量。

    平台：  
        ``Ascend`` ``GPU`` `` CPU``
