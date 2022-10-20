mindspore.dataset.audio
=======================

.. include:: dataset_audio/mindspore.dataset.audio.rst

数据增强操作可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式一般用于处理数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html#数据处理pipeline介绍>`_。
- Eager模式一般用于零散样本，音频预处理举例如下：

  .. code-block::

      import numpy as np
      import mindspore.dataset.audio as audio
      from mindspore.dataset.audio import ResampleMethod

      # 音频输入
      waveform = np.random.random([1, 30])

      # 增强操作
      resample_op = audio.Resample(orig_freq=48000, new_freq=16000,
                                   resample_method=ResampleMethod.SINC_INTERPOLATION,
                                   lowpass_filter_width=6, rolloff=0.99, beta=None)
      waveform_resampled = resample_op(waveform)
      print("waveform reampled: {}".format(waveform_resampled), flush=True)

变换
-----

.. mscnautosummary::
    :toctree: dataset_audio
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.audio.AllpassBiquad
    mindspore.dataset.audio.AmplitudeToDB
    mindspore.dataset.audio.Angle
    mindspore.dataset.audio.BandBiquad
    mindspore.dataset.audio.BandpassBiquad
    mindspore.dataset.audio.BandrejectBiquad
    mindspore.dataset.audio.BassBiquad
    mindspore.dataset.audio.Biquad
    mindspore.dataset.audio.ComplexNorm
    mindspore.dataset.audio.ComputeDeltas
    mindspore.dataset.audio.Contrast
    mindspore.dataset.audio.DBToAmplitude
    mindspore.dataset.audio.DCShift
    mindspore.dataset.audio.DeemphBiquad
    mindspore.dataset.audio.DetectPitchFrequency
    mindspore.dataset.audio.Dither
    mindspore.dataset.audio.EqualizerBiquad
    mindspore.dataset.audio.Fade
    mindspore.dataset.audio.Flanger
    mindspore.dataset.audio.FrequencyMasking
    mindspore.dataset.audio.Gain
    mindspore.dataset.audio.GriffinLim
    mindspore.dataset.audio.HighpassBiquad
    mindspore.dataset.audio.InverseMelScale
    mindspore.dataset.audio.LFilter
    mindspore.dataset.audio.LowpassBiquad
    mindspore.dataset.audio.Magphase
    mindspore.dataset.audio.MaskAlongAxis
    mindspore.dataset.audio.MaskAlongAxisIID
    mindspore.dataset.audio.MelScale
    mindspore.dataset.audio.MuLawDecoding
    mindspore.dataset.audio.MuLawEncoding
    mindspore.dataset.audio.Overdrive
    mindspore.dataset.audio.Phaser
    mindspore.dataset.audio.PhaseVocoder
    mindspore.dataset.audio.Resample
    mindspore.dataset.audio.RiaaBiquad
    mindspore.dataset.audio.SlidingWindowCmn
    mindspore.dataset.audio.SpectralCentroid
    mindspore.dataset.audio.Spectrogram
    mindspore.dataset.audio.TimeMasking
    mindspore.dataset.audio.TimeStretch
    mindspore.dataset.audio.TrebleBiquad
    mindspore.dataset.audio.Vad
    mindspore.dataset.audio.Vol

工具
-----

.. mscnautosummary::
    :toctree: dataset_audio
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.audio.BorderType
    mindspore.dataset.audio.DensityFunction
    mindspore.dataset.audio.FadeShape
    mindspore.dataset.audio.GainType
    mindspore.dataset.audio.Interpolation
    mindspore.dataset.audio.MelType
    mindspore.dataset.audio.Modulation
    mindspore.dataset.audio.NormMode
    mindspore.dataset.audio.NormType
    mindspore.dataset.audio.ResampleMethod
    mindspore.dataset.audio.ScaleType
    mindspore.dataset.audio.WindowType
    mindspore.dataset.audio.create_dct
    mindspore.dataset.audio.melscale_fbanks
