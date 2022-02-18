mindspore.dataset.vision.py_transforms.HsvToRgb
===============================================

.. py:class:: mindspore.dataset.vision.py_transforms.HsvToRgb(is_hwc=False)

    将输入的HSV格式numpy.ndarray图像转换为RGB格式。

    **参数：**
    
    - **is_hwc** (bool) - 若为True，表示输入图像的维度为(H, W, C)或(N, H, W, C)；否则为(C, H, W)或(N, C, H, W)。默认值：False。
