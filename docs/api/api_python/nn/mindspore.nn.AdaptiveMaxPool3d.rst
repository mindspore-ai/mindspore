mindspore.nn.AdaptiveMaxPool3d
==============================

.. py:class:: mindspore.nn.AdaptiveMaxPool3d(output_size, return_indices=False)

    对输入Tensor，提供三维自适应最大池化操作。对于任何输入尺寸，输出的大小为 :math:`(D, H, W)` 。输出特征的数量与输入特征的数量相同。

    参数：
        - **output_size** (Union[int, tuple]) - 表示输出特征图的尺寸，输入可以是tuple :math:`(D, H, W)`，也可以是一个int值D来表示输出尺寸为 :math:`(D, D, D)` 。:math:`D` ， :math:`H` 和 :math:`W` 可以是int型整数或者None，其中None表示输出大小与对应的输入的大小相同。
        - **return_indices** (bool) - 如果 `return_indices` 为True，将会输出最大值对应的索引，否则不输出索引。默认为False。

    输入：
        - **input** (Tensor) - shape为 :math:`(C, D, H, W)` 或 :math:`(N，C, D, H, W)` 的Tensor，支持的数据类型包括int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64。

    输出：
        - **y** (Tensor) - Tensor，与输入 `input` 的数据类型和维度相同。
        - **argmax** (Tensor) - Tensor，最大值对应的索引，数据类型为int32，并与 `y` 的shape相同。仅当 `return_indices` 为True的时候才返回该值。 

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 的维度不是4D或者5D。
        - **TypeError** - `input` 的数据类型不是int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64其中之一。
        - **ValueError** - `output_size` 不是一个int整数或者shape为(3,)的tuple。
