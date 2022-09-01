mindspore.dataset.vision.Normalize
==================================

.. py:class:: mindspore.dataset.vision.Normalize(mean, std, is_hwc=True)

    根据均值和标准差对输入图像进行归一化。

    此处理将使用以下公式对输入图像进行归一化：output[channel] = (input[channel] - mean[channel]) / std[channel]，其中 channel 代表通道索引，channel >= 1。

    .. note:: 此操作支持通过 Offload 在 Ascend 或 GPU 平台上运行。

    参数：
        - **mean** (sequence) - 图像每个通道的均值组成的列表或元组。平均值必须在 [0.0, 255.0] 范围内。
        - **std** (sequence) - 图像每个通道的标准差组成的列表或元组。标准差值必须在 (0.0, 255.0] 范围内。
        - **is_hwc** (bool, 可选) - 表示输入图像是否为HWC格式，True为HWC格式，False为CHW格式。默认值：True。

    异常：
        - **TypeError** - 如果 `mean` 不是sequence类型。
        - **TypeError** - 如果 `std` 不是sequence类型。
        - **TypeError** - 如果 `is_hwc` 不是bool类型。
        - **ValueError** - 如果 `mean` 不在 [0.0, 255.0] 范围内。
        - **ValueError** - 如果 `std` 不在 (0.0, 255.0] 范围内。
        - **RuntimeError** - 如果给定的tensor format不是<H, W>或<...,H, W, C>。
