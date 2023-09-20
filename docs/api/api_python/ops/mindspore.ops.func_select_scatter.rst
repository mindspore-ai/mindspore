mindspore.ops.select_scatter
============================

.. py:function:: mindspore.ops.select_scatter(input, src, axis, index)

    将 `src` 中的值散布到 `input` 指定维度 `axis` 的指定位置 `index` 上。

    参数：
        - **input** (Tensor) - 目标Tensor。
        - **src** (Tensor) - 源Tensor。
        - **axis** (int) - 要嵌入的 `input` 的维度。
        - **index** (int) - 在指定维度上散布的位置。

    返回：
        嵌入后的Tensor，与 `input` 有相同的shape和类型。

    异常：
        - **ValueError** - `src` 的shape与散布在 `input` 上的shape不一样。
        - **TypeError** - 如果 `input` 不是一个Tensor。
        - **TypeError** - 如果 `src` 不是一个Tensor。
        - **TypeError** - 如果 `axis` 或 `index` 不是整数。
