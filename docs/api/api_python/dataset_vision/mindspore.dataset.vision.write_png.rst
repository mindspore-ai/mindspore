mindspore.dataset.vision.write_png
==================================

.. py:function:: mindspore.dataset.vision.write_png(filename, image, compression_level=6)

    将图像数据保存为PNG文件。

    参数：
        - **filename** (str) - 要写入的文件的路径。
        - **image** (Union[numpy.ndarray, mindspore.Tensor]) - 要写入的图像数据。
        - **compression_level** (int, 可选) - 生成PNG文件的压缩级别，取值范围为[0, 9]。默认值： ``6``。

    异常：
        - **TypeError** - 如果 `filename` 不是str类型。
        - **TypeError** - 如果 `image` 不是numpy.ndarray或mindspore.Tensor类型。
        - **TypeError** - 如果 `compression_level` 不是int类型。
        - **RuntimeError** - 如果 `filename` 不存在或不是普通文件。
        - **RuntimeError** - 如果 `image` 的数据类型不是uint8类型。
        - **RuntimeError** - 如果 `image` 的shape不是 <H, W> 或 <H, W, 1> 或 <H, W, 3>。
        - **RuntimeError** - 如果 `compression_level` 小于0或大于9。
