mindspore.ops.chunk
====================

.. py:function:: mindspore.ops.chunk(x, chunks, axis=0)

    根据指定的轴将输入Tensor切分成块。

    参数：
        - **x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **chunks** (int) - 要返回的块数。
        - **axis** (int) - 指定分割轴。默认值：0。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `axis` 不是int类型。
        - **ValueError** - 参数 `axis` 超出 :math:`(-x.dim, x.dim)` 范围。
        - **TypeError** - `chunks` 不是int。
        - **ValueError** - 参数 `chunks` 不是正数。
