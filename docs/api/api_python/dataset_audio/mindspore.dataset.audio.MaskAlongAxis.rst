mindspore.dataset.audio.MaskAlongAxis
=====================================

.. py:class:: mindspore.dataset.audio.MaskAlongAxis(mask_start, mask_width, mask_value, axis)

    对音频波形应用掩码。掩码的起始和长度由 `[mask_start, mask_start + mask_width)` 决定。

    参数：
        - **mask_start** (int) - 掩码的起始位置，必须是非负的。
        - **mask_width** (int) - 掩码的宽度，必须是大于0。
        - **mask_value** (float) - 填充到掩码区间的值。
        - **axis** (int) - 要应用掩码的轴（ ``1`` 表示频率， ``2`` 表示时间）。

    异常：
        - **ValueError** - `mask_start` 参数值错误（小于0）。
        - **ValueError** - `mask_width` 参数值错误（小于1）。
        - **ValueError** - `axis` 参数类型错误或者值错误，不属于 [1, 2]。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
