mindspore.dataset.vision.SliceMode
==================================

.. py:class:: mindspore.dataset.vision.SliceMode()
	
    Tensor切片方式枚举类。

    可选枚举值为：SliceMode.PAD、SliceMode.DROP。

    - **SliceMode.PAD** - 当Tensor无法进行整数切分时，对剩余部分进行填充。
    - **SliceMode.DROP** - 当Tensor无法进行整数切分时，对剩余部分进行丢弃。
