mindspore.dataset.audio.transforms.AmplitudeToDB
=================================================

.. py:class:: mindspore.dataset.audio.transforms.AmplitudeToDB(stype=<ScaleType.POWER: 'power'>, ref_value=1.0, amin=1e-10, top_db=80.0)

    将输入音频从振幅/功率标度转换为分贝标度。

    **参数：**

    - **stype** ( :class:`mindspore.dataset.audio.utils.ScaleType` , optional) - 输入音频的原始标度（默认值为ScaleType.POWER）。取值可为ScaleType.MAGNITUDE或ScaleType.POWER。
    - **ref_value** (float, optional) - 系数参考值，用于计算分贝系数 `db_multiplier` ， 
    
       :math:`db\_multiplier = Log10(max(ref\_value, amin))`。
       
    - **amin** (float, optional) - 波形取值下界，低于该值的波形将会被裁切。取值必须大于0。
    - **top_db** (float, optional) - 最小负截止分贝值，建议的取值为80.0（默认值为80.0）。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.random.random([1, 400//2+1, 30])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.AmplitudeToDB(stype=ScaleType.POWER)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
