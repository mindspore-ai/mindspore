mindspore.dataset.vision.CutMixBatch
=================================================

.. py:class:: mindspore.dataset.vision.CutMixBatch(image_batch_format, alpha=1.0, prob=1.0)

    对输入批次的图像和标注应用剪切混合转换。
    请注意，在调用此操作符之前，您需要将标注制作为 one-hot 格式并进行批处理。

    参数：
        - **image_batch_format** (:class:`~.vision.ImageBatchFormat`) - 图像批处理输出格式。可以是 ``ImageBatchFormat.NHWC`` 或 ``ImageBatchFormat.NCHW`` 。
        - **alpha** (float, 可选) - β分布的超参数，必须大于0。默认值： ``1.0`` 。
        - **prob** (float, 可选) - 对每个图像应用剪切混合处理的概率，取值范围：[0.0, 1.0]。默认值： ``1.0`` 。

    异常：
        - **TypeError** - 如果 `image_batch_format` 不是 :class:`mindspore.dataset.vision.ImageBatchFormat` 的类型。
        - **TypeError** - 如果 `alpha` 不是float类型。
        - **TypeError** - 如果 `prob` 不是 float 类型。
        - **ValueError** - 如果 `alpha` 小于或等于 0。
        - **ValueError** - 如果 `prob` 不在 [0.0, 1.0] 范围内。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
