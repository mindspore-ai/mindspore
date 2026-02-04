
.. py:function:: mindspore.utils.dlpack.from_dlpack(dlpack)

    将DLPack对象转换为MindSpore Tensor。

    此函数允许从其他支持DLPack的深度学习框架共享张量数据。
    数据不会被复制，返回的MindSpore Tensor与源张量共享内存。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **dlpack** (PyCapsule) - 要转换的DLPack对象，它是一个包含指向 `DLManagedTensor` 的指针的capsule。

    返回：
        Tensor, 与DLPack对象共享内存的MindSpore Tensor。

.. py:function:: mindspore.utils.dlpack.to_dlpack(tensor)

    将MindSpore的Tensor转换为DLPack对象。

    DLPack格式是在不同深度学习框架之间共享张量数据的标准。
    返回的DLPack对象是一个Python capsule，可以被其他支持DLPack的库使用。
    该capsule包含一个指向 `DLManagedTensor` 结构的指针。DLPack对象的使用者负责释放内存。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **tensor** (Tensor) - 要转换的MindSpore Tensor。

    返回：
        PyCapsule, 一个可以被其他库使用的DLPack对象。
