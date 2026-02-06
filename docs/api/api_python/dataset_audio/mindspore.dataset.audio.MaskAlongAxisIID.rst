mindspore.dataset.audio.MaskAlongAxisIID
========================================

.. py:class:: mindspore.dataset.audio.MaskAlongAxisIID(mask_param, mask_value, axis)

    对音频波形沿 `axis` 轴应用掩码。掩码的起始和长度由 `[mask_start, mask_start + mask_width)` 决定，其中 `mask_width` 从 `uniform[0, mask_param]` 中均匀采样，`mask_start` 从 `uniform[0, max_length - mask_width]` 中均匀采样，
    `max_length` 是频谱图中特定轴的列数。

    参数：
        - **mask_param** (int) - 要屏蔽的列数，将从[0, mask_param]均匀采样，必须是非负数。
        - **mask_value** (float) - 填充到掩码区间的值。
        - **axis** (int) - 要应用掩码的轴， ``1`` 表示频率， ``2`` 表示时间。

    异常：
        - **TypeError** - 当 `mask_param` 的类型不为int。
        - **ValueError** - 当 `mask_param` 为负数。
        - **TypeError** - 当 `mask_value` 的类型不为float。
        - **TypeError** - 当 `axis` 的类型不为int。
        - **ValueError** - 当 `axis` 取值不在[1, 2]范围内。
        - **RuntimeError** - 当输入音频的shape不为<..., freq, time>。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
