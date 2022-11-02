mindspore.dataset.text.SlidingWindow
====================================

.. py:class:: mindspore.dataset.text.SlidingWindow(width, axis=0)

    在输入数据的某个维度上进行滑窗切分处理，当前仅支持处理1-D的Tensor。

    参数：
        - **width** (int) - 窗口的宽度，它必须是整数并且大于零。
        - **axis** (int, 可选) - 计算滑动窗口的轴。默认值：0。

    异常：
        - **TypeError** - 参数 `width` 的类型不为int。
        - **ValueError** - 参数 `width` 的值不为正数。
        - **TypeError** - 参数 `axis` 的类型不为int。
