mindspore.ops.ifftshift
=================================

.. py:function:: mindspore.ops.ifftshift(input, dim=None)

    :func:`mindspore.ops.fftshift` 的逆运算。

    .. note::
        - `ifftshift` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `ifftshift` 尚不支持Windows平台.

    参数：
        - **input** (Tensor) - 输入的Tensor。
        - **dim** (Union[int, list(int), tuple(int)], 可选) - 指定需移动维度的轴。默认值： ``None`` ，将移动所有轴。

    返回：
        Tensor，与 `input` 的数据类型和shape相同。


    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `dim` 不是上述支持的类型。
        - **ValueError** - 如果 `dim` 中的值超出： :math:`[-input.ndim, -input.ndim)` 范围。
        
