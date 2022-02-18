mindspore.dataset.vision.py_transforms.RandomHorizontalFlip
===========================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomHorizontalFlip(prob=0.5)

    按照指定的概率随机水平翻转输入的PIL图像。

    **参数：**

    - **prob** (float，可选) - 执行水平翻转的概率，默认值：0.5。

    **异常：**

    - **TypeError** - 当 `prob` 的类型不为浮点型。
    - **ValueError** - 当 `prob` 取值不在[0, 1]范围内。
    - **RuntimeError** - 当输入图像的shape不为<H, W>或<H, W, C>。
