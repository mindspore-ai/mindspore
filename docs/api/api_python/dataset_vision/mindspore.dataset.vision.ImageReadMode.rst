mindspore.dataset.vision.ImageReadMode
======================================

.. py:class:: mindspore.dataset.vision.ImageReadMode

    图像文件读取方式枚举类。

    可选枚举值为： ``ImageReadMode.UNCHANGED`` 、 ``ImageReadMode.GRAYSCALE`` 、 ``ImageReadMode.COLOR`` 。

    - **ImageReadMode.UNCHANGED** - 按照图像原始格式读取。
    - **ImageReadMode.GRAYSCALE** - 读取并转为单通道灰度数据。
    - **ImageReadMode.COLOR** - 读取并换为3通道RGB彩色数据。
