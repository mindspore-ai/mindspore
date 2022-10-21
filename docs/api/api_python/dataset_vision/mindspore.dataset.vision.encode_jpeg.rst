mindspore.dataset.vision.encode_jpeg
====================================

.. py:function:: mindspore.dataset.vision.encode_jpeg(image, quality=75)

    将输入的图像编码为JPEG数据。

    参数：
        - **image** (Union[numpy.ndarray, mindspore.Tensor]) - 编码的图像。
        - **quality** (int, 可选) - 生成的JPEG数据的质量，从1到100。默认值75。

    返回：
        - numpy:ndarray, 一维uint8类型数据。

    异常：
        - **TypeError** - 如果 `image` 不是numpy.ndarray或mindspore.Tensor类型。
        - **TypeError** - 如果 `quality` 不是int类型。
        - **RuntimeError** - 如果 `image` 的数据类型不是uint8类型。
        - **RuntimeError** - 如果 `image` 的shape不是 <H, W> 或 <H, W, 1> 或 <H, W, 3>。
        - **RuntimeError** - 如果 `quality` 小于1或大于100。
