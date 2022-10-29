mindspore.dataset.vision.RandomGrayscale
========================================

.. py:class:: mindspore.dataset.vision.RandomGrayscale(prob=0.1)

    按照指定的概率将输入PIL图像转换为灰度图。

    参数：
        - **prob** (float，可选) - 执行灰度转换的概率，取值范围：[0.0, 1.0]。默认值：0.1。

    异常：
        - **TypeError** - 当 `prob` 的类型不为float。
        - **ValueError** - 当 `prob` 取值不在[0.0, 1.0]范围内。
