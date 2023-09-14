mindspore.dataset.vision.RandomHorizontalFlipWithBBox
=====================================================

.. py:class:: mindspore.dataset.vision.RandomHorizontalFlipWithBBox(prob=0.5)

    按给定的概率，对输入图像及其边界框进行随机水平翻转。

    参数：
        - **prob**  (float, 可选) - 图像被翻转的概率，取值范围：[0.0, 1.0]。默认值： ``0.5`` 。

    异常：
        - **TypeError** - 如果 `prob` 不是float类型。
        - **ValueError** - 如果 `prob` 不在 [0.0, 1.0] 范围内。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
