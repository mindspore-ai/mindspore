mindspore.mint.chunk
====================

.. py:function:: mindspore.mint.chunk(input, chunks, dim=0)

    沿着指定轴 `dim` 将输入Tensor切分成 `chunks` 个sub-tensor。

    .. note::
        此函数返回的数量可能小于通过 `chunks` 指定的数量。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 被切分的Tensor。
        - **chunks** (int) - 要切分的sub-tensor数量。
        - **dim** (int，可选) - 指定需要分割的维度。默认值： ``0`` 。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `chunks` 不是int。
        - **TypeError** - `dim` 不是int类型。
        - **ValueError** - 参数 `dim` 超出 :math:`[-input.ndim, input.ndim)` 范围。
        - **ValueError** - 参数 `chunks` 不是正数。
