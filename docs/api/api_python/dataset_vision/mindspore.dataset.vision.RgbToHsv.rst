mindspore.dataset.vision.RgbToHsv
=================================

.. py:class:: mindspore.dataset.vision.RgbToHsv(is_hwc=False)

    将输入的RGB格式numpy.ndarray图像转换为HSV格式。

    参数：
        - **is_hwc** (bool) - 若为 ``True`` ，表示输入图像的shape为<H, W, C>或<N, H, W, C>；否则为<C, H, W>或<N, C, H, W>。默认值： ``False`` 。

    异常：
        - **TypeError** - 当 `is_hwc` 的类型不为bool。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
