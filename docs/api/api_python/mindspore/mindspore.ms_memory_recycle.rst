mindspore.ms_memory_recycle
================================

.. py:function:: mindspore.ms_memory_recycle()

    回收MindSpore使用的内存。

    在一个进程中训练多个神经网络模型时，MindSpore使用的内存非常大，这是因为MindSpore为每个模型缓存了运行时的内存。

    为了回收这些缓存的内存，用户可以在训练一个模型后调用这个函数。
