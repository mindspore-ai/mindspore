mindspore.dataset.vision.write_jpeg
===================================

.. py:function:: mindspore.dataset.vision.write_jpeg(filename, image, quality=75)

    将图像数据保存为JPEG文件。

    参数：
        - **filename** (str) - 要写入的文件的路径。
        - **image** (Union[numpy.ndarray, mindspore.Tensor]) - 要写入的图像数据。
        - **quality** (int, 可选) - 生成的JPEG文件的质量，取值范围为[1, 100]。默认值:  ``75`` 。

    异常：
        - **TypeError** - 如果 `filename` 不是str类型。
        - **TypeError** - 如果 `image` 不是numpy.ndarray或mindspore.Tensor类型。
        - **TypeError** - 如果 `quality` 不是int类型。
        - **RuntimeError** - 如果 `filename` 不存在或不是普通文件。
        - **RuntimeError** - 如果 `image` 的数据类型不是uint8类型。
        - **RuntimeError** - 如果 `image` 的shape不是 <H, W> 或 <H, W, 1> 或 <H, W, 3>。
        - **RuntimeError** - 如果 `quality` 小于1或大于100。
