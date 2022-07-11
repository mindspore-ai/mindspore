mindspore.dataset.vision.Decode
===============================

.. py:class:: mindspore.dataset.vision.Decode(to_pil=False)

    将输入的压缩图像解码为RGB格式。当前支持的图片类型：JPEG, BMP, PNG, TIFF, GIF(需要指定 `to_pil=True`), WEBP(需要指定 `to_pil=True`)。

    参数：
        - **to_pil** (bool，可选) - 是否将图像解码为PIL数据类型。若为True，图像将被解码为PIL数据类型，否则解码为NumPy数据类型。默认值：False。

    异常：
        - **RuntimeError** - 如果输入图像不是一维序列。
        - **RuntimeError** - 如果输入数据不是合法的图像字节数据。
        - **RuntimeError** - 如果输入数据已经是解码的图像数据。
