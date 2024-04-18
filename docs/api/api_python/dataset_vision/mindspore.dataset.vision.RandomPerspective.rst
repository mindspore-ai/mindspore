mindspore.dataset.vision.RandomPerspective
==========================================

.. py:class:: mindspore.dataset.vision.RandomPerspective(distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC)

    按照指定的概率对输入PIL图像进行透视变换。

    参数：
        - **distortion_scale** (float，可选) - 失真程度，取值范围为[0.0, 1.0]。默认值： ``0.5`` 。
        - **prob** (float，可选) - 执行透视变换的概率，取值范围：[0.0, 1.0]。默认值： ``0.5`` 。
        - **interpolation** (:class:`~.vision.Inter`，可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.BICUBIC``。

    异常：
        - **TypeError** - 当 `distortion_scale` 的类型不为float。
        - **TypeError** - 当 `prob` 的类型不为float。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **ValueError** - 当 `distortion_scale` 取值不在[0.0, 1.0]范围内。
        - **ValueError** - 当 `prob` 取值不在[0.0, 1.0]范围内。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
