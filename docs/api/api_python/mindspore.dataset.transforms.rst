mindspore.dataset.transforms
============================

通用
-----

此模块用于通用数据增强，其中一部分增强操作是用C++实现的，具有较好的高性能，另一部分是基于Python实现，使用了NumPy模块作为支持。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- PyTensorOperation，所有Python实现的数据处理操作的基类。

变换
^^^^^

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.Compose
    mindspore.dataset.transforms.Concatenate
    mindspore.dataset.transforms.Duplicate
    mindspore.dataset.transforms.Fill
    mindspore.dataset.transforms.Mask
    mindspore.dataset.transforms.OneHot
    mindspore.dataset.transforms.PadEnd
    mindspore.dataset.transforms.RandomApply
    mindspore.dataset.transforms.RandomChoice
    mindspore.dataset.transforms.RandomOrder
    mindspore.dataset.transforms.Slice
    mindspore.dataset.transforms.TypeCast
    mindspore.dataset.transforms.Unique

工具
^^^^^

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.Relational

视觉
-----

此模块用于图像数据增强，其中有一部分增强是基于C++ OpenCV实现的，具有较好的性能，而另一部分是基于Python Pillow实现的。

API样例中常用的导入模块如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore.dataset.vision.utils as utils

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- ImageTensorOperation，所有图像类数据处理操作的基类，继承自TensorOperation。
- PyTensorOperation，所有Python实现的数据处理操作的基类。

数据增强操作可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式用于流式处理大型数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.loading.html#数据处理pipeline介绍>`_ 。
- Eager模式用于函数调用方式处理样本，示例可参考 `轻量化数据处理 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/eager.html>`_ 。

样例库
^^^^^^

快速上手使用视觉类变换的API，跳转参考 `视觉变换样例库 <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_ 。
此指南中展示了多个变换API的用法，以及输入输出结果。

变换
^^^^^

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.AdjustBrightness
    mindspore.dataset.vision.AdjustContrast
    mindspore.dataset.vision.AdjustGamma
    mindspore.dataset.vision.AdjustHue
    mindspore.dataset.vision.AdjustSaturation
    mindspore.dataset.vision.AdjustSharpness
    mindspore.dataset.vision.Affine
    mindspore.dataset.vision.AutoAugment
    mindspore.dataset.vision.AutoContrast
    mindspore.dataset.vision.BoundingBoxAugment
    mindspore.dataset.vision.CenterCrop
    mindspore.dataset.vision.ConvertColor
    mindspore.dataset.vision.Crop
    mindspore.dataset.vision.CutMixBatch
    mindspore.dataset.vision.CutOut
    mindspore.dataset.vision.Decode
    mindspore.dataset.vision.DecodeVideo
    mindspore.dataset.vision.Equalize
    mindspore.dataset.vision.Erase
    mindspore.dataset.vision.FiveCrop
    mindspore.dataset.vision.GaussianBlur
    mindspore.dataset.vision.Grayscale
    mindspore.dataset.vision.HorizontalFlip
    mindspore.dataset.vision.HsvToRgb
    mindspore.dataset.vision.HWC2CHW
    mindspore.dataset.vision.Invert
    mindspore.dataset.vision.LinearTransformation
    mindspore.dataset.vision.MixUp
    mindspore.dataset.vision.MixUpBatch
    mindspore.dataset.vision.Normalize
    mindspore.dataset.vision.NormalizePad
    mindspore.dataset.vision.Pad
    mindspore.dataset.vision.PadToSize
    mindspore.dataset.vision.Perspective
    mindspore.dataset.vision.Posterize
    mindspore.dataset.vision.RandAugment
    mindspore.dataset.vision.RandomAdjustSharpness
    mindspore.dataset.vision.RandomAffine
    mindspore.dataset.vision.RandomAutoContrast
    mindspore.dataset.vision.RandomColor
    mindspore.dataset.vision.RandomColorAdjust
    mindspore.dataset.vision.RandomCrop
    mindspore.dataset.vision.RandomCropDecodeResize
    mindspore.dataset.vision.RandomCropWithBBox
    mindspore.dataset.vision.RandomEqualize
    mindspore.dataset.vision.RandomErasing
    mindspore.dataset.vision.RandomGrayscale
    mindspore.dataset.vision.RandomHorizontalFlip
    mindspore.dataset.vision.RandomHorizontalFlipWithBBox
    mindspore.dataset.vision.RandomInvert
    mindspore.dataset.vision.RandomLighting
    mindspore.dataset.vision.RandomPerspective
    mindspore.dataset.vision.RandomPosterize
    mindspore.dataset.vision.RandomResizedCrop
    mindspore.dataset.vision.RandomResizedCropWithBBox
    mindspore.dataset.vision.RandomResize
    mindspore.dataset.vision.RandomResizeWithBBox
    mindspore.dataset.vision.RandomRotation
    mindspore.dataset.vision.RandomSelectSubpolicy
    mindspore.dataset.vision.RandomSharpness
    mindspore.dataset.vision.RandomSolarize
    mindspore.dataset.vision.RandomVerticalFlip
    mindspore.dataset.vision.RandomVerticalFlipWithBBox
    mindspore.dataset.vision.Rescale
    mindspore.dataset.vision.Resize
    mindspore.dataset.vision.ResizedCrop
    mindspore.dataset.vision.ResizeWithBBox
    mindspore.dataset.vision.RgbToHsv
    mindspore.dataset.vision.Rotate
    mindspore.dataset.vision.SlicePatches
    mindspore.dataset.vision.Solarize
    mindspore.dataset.vision.TenCrop
    mindspore.dataset.vision.ToNumpy
    mindspore.dataset.vision.ToPIL
    mindspore.dataset.vision.ToTensor
    mindspore.dataset.vision.ToType
    mindspore.dataset.vision.TrivialAugmentWide
    mindspore.dataset.vision.UniformAugment
    mindspore.dataset.vision.VerticalFlip

工具
^^^^^

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.AutoAugmentPolicy
    mindspore.dataset.vision.Border
    mindspore.dataset.vision.ConvertMode
    mindspore.dataset.vision.ImageBatchFormat
    mindspore.dataset.vision.ImageReadMode
    mindspore.dataset.vision.Inter
    mindspore.dataset.vision.SliceMode
    mindspore.dataset.vision.VideoDecoder
    mindspore.dataset.vision.encode_jpeg
    mindspore.dataset.vision.encode_png
    mindspore.dataset.vision.get_image_num_channels
    mindspore.dataset.vision.get_image_size
    mindspore.dataset.vision.read_file
    mindspore.dataset.vision.read_image
    mindspore.dataset.vision.read_video
    mindspore.dataset.vision.read_video_timestamps
    mindspore.dataset.vision.write_file
    mindspore.dataset.vision.write_jpeg
    mindspore.dataset.vision.write_png

文本
-----

此模块用于文本数据增强，包括 `transforms` 和 `utils` 两个子模块。

`transforms` 是一个高性能文本数据增强模块，支持常见的文本数据增强处理。

`utils` 提供了一些文本处理的工具方法。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.dataset as ds
    from mindspore.dataset import text

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- TextTensorOperation，所有文本数据处理操作的基类，派生自TensorOperation。

数据增强操作可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式用于流式处理大型数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.loading.html#数据处理pipeline介绍>`_ 。
- Eager模式用于函数调用方式处理样本，示例可参考 `轻量化数据处理 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/eager.html>`_ 。

样例库
^^^^^^

快速上手使用文本变换的API，跳转参考 `文本变换样例库 <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_ 。
此指南中展示了多个变换API的用法，以及输入输出结果。

变换
^^^^^

.. mscnnoteautosummary::
    :toctree: dataset_text
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.text.AddToken
    mindspore.dataset.text.BasicTokenizer
    mindspore.dataset.text.BertTokenizer
    mindspore.dataset.text.CaseFold
    mindspore.dataset.text.FilterWikipediaXML
    mindspore.dataset.text.JiebaTokenizer
    mindspore.dataset.text.Lookup
    mindspore.dataset.text.Ngram
    mindspore.dataset.text.NormalizeUTF8
    mindspore.dataset.text.PythonTokenizer
    mindspore.dataset.text.RegexReplace
    mindspore.dataset.text.RegexTokenizer
    mindspore.dataset.text.SentencePieceTokenizer
    mindspore.dataset.text.SlidingWindow
    mindspore.dataset.text.ToNumber
    mindspore.dataset.text.ToVectors
    mindspore.dataset.text.Truncate
    mindspore.dataset.text.TruncateSequencePair
    mindspore.dataset.text.UnicodeCharTokenizer
    mindspore.dataset.text.UnicodeScriptTokenizer
    mindspore.dataset.text.WhitespaceTokenizer
    mindspore.dataset.text.WordpieceTokenizer


工具
^^^^^

.. mscnnoteautosummary::
    :toctree: dataset_text
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.text.CharNGram
    mindspore.dataset.text.FastText
    mindspore.dataset.text.GloVe
    mindspore.dataset.text.JiebaMode
    mindspore.dataset.text.NormalizeForm
    mindspore.dataset.text.SentencePieceModel
    mindspore.dataset.text.SentencePieceVocab
    mindspore.dataset.text.SPieceTokenizerLoadType
    mindspore.dataset.text.SPieceTokenizerOutType
    mindspore.dataset.text.Vectors
    mindspore.dataset.text.Vocab
    mindspore.dataset.text.to_bytes
    mindspore.dataset.text.to_str

音频
-----

.. include:: dataset_audio/mindspore.dataset.audio.rst

数据增强操作可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式用于流式处理大型数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.loading.html#数据处理pipeline介绍>`_ 。
- Eager模式用于函数调用方式处理样本，示例可参考 `轻量化数据处理 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/eager.html>`_ 。

样例库
^^^^^^

快速上手使用音频变换的API，跳转参考 `音频变换样例库 <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_ 。
此指南中展示了多个变换API的用法，以及输入输出结果。

变换
^^^^^

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
    mindspore.dataset.audio.Filtfilt
    mindspore.dataset.audio.Flanger
    mindspore.dataset.audio.FrequencyMasking
    mindspore.dataset.audio.Gain
    mindspore.dataset.audio.GriffinLim
    mindspore.dataset.audio.HighpassBiquad
    mindspore.dataset.audio.InverseMelScale
    mindspore.dataset.audio.InverseSpectrogram
    mindspore.dataset.audio.LFCC
    mindspore.dataset.audio.LFilter
    mindspore.dataset.audio.LowpassBiquad
    mindspore.dataset.audio.Magphase
    mindspore.dataset.audio.MaskAlongAxis
    mindspore.dataset.audio.MaskAlongAxisIID
    mindspore.dataset.audio.MelScale
    mindspore.dataset.audio.MelSpectrogram
    mindspore.dataset.audio.MFCC
    mindspore.dataset.audio.MuLawDecoding
    mindspore.dataset.audio.MuLawEncoding
    mindspore.dataset.audio.Overdrive
    mindspore.dataset.audio.Phaser
    mindspore.dataset.audio.PhaseVocoder
    mindspore.dataset.audio.PitchShift
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
^^^^^^

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
    mindspore.dataset.audio.linear_fbanks
    mindspore.dataset.audio.melscale_fbanks
