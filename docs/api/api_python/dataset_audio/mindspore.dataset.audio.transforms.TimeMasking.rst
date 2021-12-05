mindspore.dataset.audio.transforms.TimeMasking
=================================================

.. py:class:: mindspore.dataset.audio.transforms.TimeMasking(iid_masks=False, time_mask_param=0, mask_start=0, mask_value=0.0)

    给音频波形添加时域掩码。

    **参数：**

    - **iid_masks** (bool, optional) - 是否添加随机掩码（默认为False）。
    - **time_mask_param** (int): 当 `iid_masks` 为True时，掩码长度将从[0, time_mask_param]中均匀采样；当 `iid_masks` 为False时，使用该值作为掩码的长度。取值范围为[0, time_length]，其中 `time_length` 为波形在时域的长度（默认为0）。
    - **mask_start** (int) - 添加掩码的起始位置，只有当 `iid_masks` 为True时，该值才会生效。取值范围为[0, time_length - time_mask_param]，其中 `time_length` 为波形在时域的长度（默认为0）。
    - **mask_value** (double) - 添加掩码的取值（默认为0.0）。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.random.random([1, 3, 2])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.TimeMasking(time_mask_param=1)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
