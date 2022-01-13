mindspore.dataset.audio.transforms.FrequencyMasking
===================================================

.. py:class:: mindspore.dataset.audio.transforms.FrequencyMasking(iid_masks=False, frequency_mask_param=0, mask_start=0, mask_value=0.0)

    给音频波形施加频域掩码。

    .. note:: 待处理音频维度需为(..., freq, time)。

    **参数：**

    - **iid_masks** (bool, 可选) - 是否施加随机掩码，默认值：False。
    - **freq_mask_param** (int, 可选) - 当 `iid_masks` 为True时，掩码长度将从[0, freq_mask_param]中均匀采样；当 `iid_masks` 为False时，直接使用该值作为掩码长度。取值范围为[0, freq_length]，其中 `freq_length` 为音频波形在频域的长度，默认值：0。
    - **mask_start** (int, 可选) - 添加掩码的起始位置，只有当 `iid_masks` 为True时，该值才会生效。取值范围为[0, freq_length - freq_mask_param]，其中 `freq_length` 为音频波形在频域的长度，默认值：0。
    - **mask_value** (float, 可选) - 掩码填充值，默认值：0.0。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.random.random([1, 3, 2])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.FrequencyMasking(frequency_mask_param=1)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])

    .. image:: api_img/dataset/frequency_masking_original.png

    .. image:: api_img/dataset/frequency_masking.png

