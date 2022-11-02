mindspore.dataset.vision.get_image_num_channels
================================================

.. py:function:: mindspore.dataset.vision.get_image_num_channels(image)

    获取输入图像通道数。

    参数：
        - **image** (Union[numpy.ndarray, PIL.Image.Image]) - 用于获取通道数的图像。

    返回：
        int，输入图像通道数。

    异常：
        - **RuntimeError** - `image` 参数的维度小于2。
        - **TypeError** - `image` 参数的类型既不是 np.ndarray，也不是 PIL Image。
