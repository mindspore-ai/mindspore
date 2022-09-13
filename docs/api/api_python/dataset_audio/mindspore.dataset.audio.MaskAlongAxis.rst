mindspore.dataset.audio.MaskAlongAxis
=====================================

.. py:class:: mindspore.dataset.audio.MaskAlongAxis(mask_start, mask_width, mask_value, axis)

    对音频波形应用掩码。掩码的起始和长度由 `[mask_start, mask_start + mask_width)` 决定。

    参数：
        - **mask_start** (int) - 掩码的起始位置，必须是非负的。
        - **mask_width** (int) - 掩码的宽度，必须是非负的。
        - **mask_value** (float) - 掩码值。
        - **axis** (int) - 要应用掩码的轴（1表示频率，2表示时间）。
