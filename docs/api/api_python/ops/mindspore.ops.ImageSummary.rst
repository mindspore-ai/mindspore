mindspore.ops.ImageSummary
==========================

.. py:class:: mindspore.ops.ImageSummary

    将图片数据放到缓冲区。

    输入：
        - **name** (str) - 输入变量的名称，不能是空字符串。
        - **value** (Tensor) - 图像数据的值，Tensor的rank必须为4。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
