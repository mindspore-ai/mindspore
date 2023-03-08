mindspore.ops.chunk
====================

.. py:function:: mindspore.ops.chunk(input, chunks, axis=0)

    沿着指定轴 `axis` 将输入Tensor切分成 `chunks` 个sub-tensor。

    .. note::
        此函数返回的数量可能小于通过 `chunks` 指定的数量!

    参数：
        - **input** (Tensor) - 被切分的Tensor。
        - **chunks** (int) - 要切分的sub-tensor数量。
        - **axis** (int) - 指定需要分割的维度。默认值：0。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `axis` 不是int类型。
        - **ValueError** - 参数 `axis` 超出 :math:`(-input.dim, input.dim)` 范围。
        - **TypeError** - `chunks` 不是int。
        - **ValueError** - 参数 `chunks` 不是正数。
