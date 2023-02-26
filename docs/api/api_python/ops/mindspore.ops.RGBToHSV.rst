mindspore.ops.RGBToHSV
=======================

.. py:class:: mindspore.ops.RGBToHSV

    将一张或多张图像从RGB颜色空间转换为HSV颜色空间。
    其中每个像素的RGB值转换为其对应的HSV值。此函数仅适用于输入像素值在[0,1]范围内的情况。

    .. note::
        输入图片的最后一维长度必须为3。

    输入：
        - **images** (Tensor) - 输入的需要转换格式的一维或者多维RGB数据Tensor，最后一维的长度必须为3。支持的数据类型有：float16、float32或float64。

    输出：
        Tensor，数据类型和shape与 `images` 相同。

    异常：
        - **TypeError** - `images` 不是Tensor或者其数据类型不是float或double。
        - **ValueError** - `images` 的rank小于1。
        - **ValueError** - `images` shape的最后一维长度不等于3。
