mindspore.dataset.vision.GaussianBlur
=====================================

.. py:class:: mindspore.dataset.vision.GaussianBlur(kernel_size, sigma=None)

    使用指定的高斯核对输入图像进行模糊处理。

    参数：
        - **kernel_size** (Union[int, Sequence[int]]) - 要使用的高斯核的大小。该值必须是正数和奇数。
          如果只提供一个整数，高斯核大小将为 (kernel_size, kernel_size)。
          如果提供了整数序列，则它必须是表示（宽度、高度）的 2 个值的序列。
        - **sigma** (Union[float, Sequence[float]], 可选) - 要使用的高斯核的标准差，该值必须是正数。默认值：None。
          如果仅提供浮点数，则 `sigma` 将为 (sigma, sigma)。
          如果提供了浮点序列，则它必须是表示（宽度、高度）的 2 个值的序列。
          如果为None， `sigma` 采用的值为 ((kernel\underline{} size - 1) * 0.5 - 1) * 0.3 + 0.8。

    异常：
        - **TypeError** - 如果 `kernel_size` 不是int或Sequence[int]类型。
        - **TypeError** - 如果 `sigma` 不是float或Sequence[float]类型。
        - **ValueError** - 如果 `kernel_size` 不是正数和奇数。
        - **ValueError** - 如果 `sigma` 不是正数。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
