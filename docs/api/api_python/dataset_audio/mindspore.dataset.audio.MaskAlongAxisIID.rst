mindspore.dataset.audio.MaskAlongAxisIID
========================================

.. py:class:: mindspore.dataset.audio.MaskAlongAxisIID(mask_param, mask_value, axis)

    对音频波形应用掩码。掩码的起始和长度由 `[mask_start, mask_start + mask_width)` 决定，其中 `mask_width` 从 `uniform[0, mask_param]` 中采样， `mask_start` 从 `uniform[0, max_length - mask_width]` 中采样，
    `max_length` 是光谱图中特定轴的列数。

    参数：
        - **mask_param** (int) - 要屏蔽的列数，将从[0, mask_param]统一采样，必须是非负数。
        - **mask_value** (float) - 掩码值。
        - **axis** (int) - 要应用掩码的轴（1表示频率，2表示时间）。

    异常：
        - **ValueError** - `mask_param` 参数值错误（小于0）。
        - **ValueError** - `axis` 参数类型错误或者值错误，不属于 [1, 2]。
