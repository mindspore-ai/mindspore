mindspore.dataset.vision.HWC2CHW
================================

.. py:class:: mindspore.dataset.vision.HWC2CHW()

    将输入图像的shape从 <H, W, C> 转换为 <C, H, W>。
    如果输入图像的shape为 <H, W> ，图像将保持不变。

    .. note:: 此操作默认通过 CPU 执行，也支持异构加速到 GPU 或 Ascend 上执行。

    异常：
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
