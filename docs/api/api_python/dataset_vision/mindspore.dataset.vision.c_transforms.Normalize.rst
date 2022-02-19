mindspore.dataset.vision.c_transforms.Normalize
===============================================

.. py:class:: mindspore.dataset.vision.c_transforms.Normalize(mean, std)

    根据均值和标准差对输入图像进行归一化。
    此处理将使用以下公式对输入图像进行归一化：output[channel] = (input[channel] - mean[channel]) / std[channel]，其中 channel 代表通道索引，channel >= 1。

    **参数：**

    - **mean**  (sequence) - 图像每个通道的均值组成的列表或元组。 平均值必须在 [0.0, 255.0] 范围内。
    - **std**  (sequence) - 图像每个通道的标准差组成的列表或元组。 标准偏差值必须在 (0.0, 255.0] 范围内。

    **异常：**

    - **TypeError** - 如果 `mean` 不是sequence类型。
    - **TypeError** - 如果 `std` 不是sequence类型。
    - **ValueError** - 如果 `mean` 不在 [0.0, 255.0] 范围内。
    - **ValueError** - 如果 `mean` 不在范围内 (0.0, 255.0]。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
