mindspore.mint.real
====================

.. py:function:: mindspore.mint.real(input) -> Tensor

    返回一个新tensor，包含输入tensor的实部。如果输入tensor是实数，则返回输入tensor不变。

    返回的tensor和输入tensor共享相同的底层存储。

    .. note::
        仅支持Pynative模式。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor，输出shape与 `input` 相同。如果输入为complex64，则输出为float32，如果输入为complex128，则输出为float64。
        其他情况，输出与输入类型相同。

    异常：
        - **ValueError** - 如果输入tensor没有存储信息。