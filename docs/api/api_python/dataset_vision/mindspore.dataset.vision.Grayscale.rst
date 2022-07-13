mindspore.dataset.vision.Grayscale
==================================

.. py:class:: mindspore.dataset.vision.Grayscale(num_output_channels=1)

    将输入PIL图像转换为灰度图。

    参数：
        - **num_output_channels** (int) - 输出灰度图的通道数，取值可为1或3。当取值为3时，返回图像各通道的像素值将相同。默认值：1。

    异常：
        - **TypeError** - 当 `num_output_channels` 的类型不为int。
        - **ValueError** - 当 `num_output_channels` 取值不为1或3。
