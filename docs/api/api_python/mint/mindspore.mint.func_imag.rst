mindspore.mint.imag
====================

.. py:function:: mindspore.mint.imag(input) -> Tensor

    返回一个新tensor，包含输入tensor的虚部。

    返回的tensor和输入tensor共享相同的底层存储。

    .. note::
        - 仅支持Pynative模式。
        - 仅支持complex64和complex128类型的tensor。

    参数：
        - **input** (Tensor) - 输入tensor。数据类型只支持complex64和complex128。

    返回：
        Tensor，输出shape与 `input` 相同。如果输入为complex64，则输出为float32，如果输入为complex128，则输出为float64。

    异常：
        - **TypeError** - 如果 `input` 的数据类型不是complex64或complex128。
        - **ValueError** - 如果输入tensor没有存储信息。