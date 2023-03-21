mindspore.dataset.vision.HsvToRgb
=================================

.. py:class:: mindspore.dataset.vision.HsvToRgb(is_hwc=False)

    将输入的HSV格式numpy.ndarray图像转换为RGB格式。

    参数：
        - **is_hwc** (bool) - 若为True，表示输入图像的shape为<H, W, C>或<N, H, W, C>；否则为<C, H, W>或<N, C, H, W>。默认值：False。

    异常：
        - **TypeError** - 当 `is_hwc` 的类型不为bool。
