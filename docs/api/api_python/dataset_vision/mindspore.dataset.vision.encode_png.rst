mindspore.dataset.vision.encode_png
===================================

.. py:function:: mindspore.dataset.vision.encode_png(image, compression_level=6)

    将输入的图像编码为PNG数据。

    参数：
        - **image** (Union[numpy.ndarray, mindspore.Tensor]) - 编码的图像。
        - **compression_level** (int, 可选) - 编码压缩因子，取值范围为[0, 9]。默认值： ``6`` 。

    返回：
        - numpy.ndarray, 一维uint8类型数据。

    异常：
        - **TypeError** - 如果 `image` 不是numpy.ndarray或mindspore.Tensor类型。
        - **TypeError** - 如果 `compression_level` 不是int类型。
        - **RuntimeError** - 如果 `image` 的数据类型不是uint8类型。
        - **RuntimeError** - 如果 `image` 的shape不是 <H, W> 或 <H, W, 1> 或 <H, W, 3>。
        - **RuntimeError** - 如果 `compression_level` 小于0或大于9。
