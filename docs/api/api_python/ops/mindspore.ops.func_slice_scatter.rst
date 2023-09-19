mindspore.ops.slice_scatter
===========================

.. py:function:: mindspore.ops.slice_scatter(input, src, axis=0, start=None, end=None, step=1)

    指定维度上对输入Tensor进行切片并将源Tensor覆盖切片结果。`input` 沿着指定维度进行切片，切片的起始位置是 `start` ，结束位置是 `end` ，步长是 `step` ，然后将 `src` 覆盖切片结果，得到输出Tensor。

    从 `begin` 指定的位置开始，根据 `size` 的shape对输入Tensor进行切片。 `begin` 表示 `input_x` 每个维度的偏移量。 `size` 表示输出Tensor的大小。

    参数：
        - **input** (Tensor) - 目标Tensor。
        - **src** (Tensor) - 源Tensor。
        - **axis** (int，可选) - 要切片的 `input` 的维度。默认值: ``0`` 。
        - **start** (int，可选) - 在指定维度中切片的开始索引。默认值: ``None`` ， `start` 为 ``0`` 。
        - **end** (int，可选) - 在指定维度中切片的结束索引。默认值: ``None`` ，`end` 是 `input` 在指定维度的长度。
        - **step** (int，可选) - 步长。默认值: ``1`` ，与下一个切片元素的距离为 ``1`` 。

    返回：
        嵌入后的Tensor，与 `input` 有相同的shape和类型。

    异常：
        - **ValueError** - `src` 的shape与 `input` 切片的shape不同。
        - **TypeError** - 如果 `input` 不是一个Tensor。
        - **TypeError** - 如果 `src` 不是一个Tensor。
        - **TypeError** - 如果 `axis` 或 `step` 不是整数。
        - **TypeError** - 如果 `start` 或 `end` 不是 ``None`` 或整数。
