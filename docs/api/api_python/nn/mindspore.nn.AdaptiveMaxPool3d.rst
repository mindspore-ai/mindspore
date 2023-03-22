mindspore.nn.AdaptiveMaxPool3d
==============================

.. py:class:: mindspore.nn.AdaptiveMaxPool3d(output_size, return_indices=False)

    对输入Tensor执行三维自适应最大池化操作。对于任何输入尺寸，输出的size为 :math:`(D, H, W)` 。

    参数：
        - **output_size** (Union[int, tuple]) - 指定输出的size。可以用一个正整数统一表示输出的深度、高度和宽度，或者用一个正整数三元组来分别表示输出的深度、高度和宽度。如果是None则表示对应维度输出和输入相同。
        - **return_indices** (bool, 可选) - 如果 `return_indices` 为True，将会输出最大值对应的索引，否则不输出索引。默认值：False。

    输入：
        - **input** (Tensor) - shape为 :math:`(C, D, H, W)` 或 :math:`(N, C, D, H, W)` 的Tensor。

    输出：
        - **y** (Tensor) - Tensor，与输入 `input` 的数据类型和维度相同。
        - **argmax** (Tensor) - Tensor，最大值对应的索引，数据类型为int32，并与 `y` 的shape相同。仅当 `return_indices` 为True的时候才返回该值。 

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 的维度不是4D或者5D。
        - **TypeError** - `input` 的数据类型不是int、uint或float。
        - **ValueError** - `output_size` 不是一个int整数或者shape为(3,)的tuple。
