mindspore.dataset.transforms
============================

通用
-----

此模块用于通用数据增强，其中一部分增强操作是用C++实现的，具有较好的高性能，另一部分是基于Python实现，使用了NumPy模块作为支持。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms

注意：旧的API导入方式已经过时且会逐步废弃，因此推荐使用上面的方式，但目前仍可按以下方式导入：

.. code-block::

    from mindspore.dataset.transforms import c_transforms
    from mindspore.dataset.transforms import py_transforms

更多详情请参考 `通用数据变换 <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/transforms.html#common-transforms>`_ 。

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

注意：旧的API导入方式已经过时且会逐步废弃，因此推荐使用上面的方式，但目前仍可按以下方式导入：

.. code-block::

    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.vision.py_transforms as py_vision
    from mindspore.dataset.transforms import c_transforms

更多详情请参考 `视觉数据变换 <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/transforms.html#vision-transforms>`_ 。

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- PyTensorOperation，所有Python实现的数据处理操作的基类。

数据增强操作可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式一般用于处理数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html#数据处理pipeline介绍>`_ 。
- Eager模式一般用于零散样本，图像预处理举例如下：

  .. code-block::

      import numpy as np
      import mindspore.dataset.vision as vision
      from PIL import Image,ImageFont,ImageDraw

      # 画圆形
      img = Image.new("RGB", (300, 300), (255, 255, 255))
      draw = ImageDraw.Draw(img)
      draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
      img.save("./1.jpg")
      with open("./1.jpg", "rb") as f:
          data = f.read()

      data_decoded = vision.Decode()(data)
      data_croped = vision.RandomCrop(size=(250, 250))(data_decoded)
      data_resized = vision.Resize(size=(224, 224))(data_croped)
      data_normalized = vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                         std=[0.229 * 255, 0.224 * 255, 0.225 * 255])(data_resized)
      data_hwc2chw = vision.HWC2CHW()(data_normalized)
      print("data: {}, shape: {}".format(data_hwc2chw, data_hwc2chw.shape), flush=True)

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
    mindspore.dataset.vision.encode_jpeg
    mindspore.dataset.vision.encode_png
    mindspore.dataset.vision.get_image_num_channels
    mindspore.dataset.vision.get_image_size
    mindspore.dataset.vision.read_file
    mindspore.dataset.vision.read_image
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

更多详情请参考 `文本数据变换 <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/transforms.html#text-transforms>`_ 。

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- TextTensorOperation，所有文本数据处理操作的基类，派生自TensorOperation。

数据增强操作可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式一般用于处理数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html#数据处理pipeline介绍>`_ 。
- Eager模式一般用于零散样本，文本预处理举例如下：

  .. code-block::

      from mindspore.dataset import text
      from mindspore.dataset.text import NormalizeForm

      # 构造词汇表
      vocab_list = {"床": 1, "前": 2, "明": 3, "月": 4, "光": 5, "疑": 6,
                    "是": 7, "地": 8, "上": 9, "霜": 10, "举": 11, "头": 12,
                    "望": 13, "低": 14, "思": 15, "故": 16, "乡": 17, "繁": 18,
                    "體": 19, "字": 20, "嘿": 21, "哈": 22, "大": 23, "笑": 24,
                    "嘻": 25, "UNK": 26}
      vocab = text.Vocab.from_dict(vocab_list)
      tokenizer_op = text.BertTokenizer(vocab=vocab, suffix_indicator='##', max_bytes_per_token=100,
                                        unknown_token='[UNK]', lower_case=False, keep_whitespace=False,
                                        normalization_form=NormalizeForm.NONE, preserve_unused_token=True,
                                        with_offsets=False)
      # 分词
      tokens = tokenizer_op("床前明月光，疑是地上霜，举头望明月，低头思故乡。")
      print("token: {}".format(tokens), flush=True)

      # 根据单词查找id
      ids = vocab.tokens_to_ids(tokens)
      print("token to id: {}".format(ids), flush=True)

      # 根据id查找单词
      tokens_from_ids = vocab.ids_to_tokens([15, 3, 7])
      print("token to id: {}".format(tokens_from_ids), flush=True)

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

- Pipeline模式一般用于处理数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html#数据处理pipeline介绍>`_ 。
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
    mindspore.dataset.audio.Resample
    mindspore.dataset.audio.PitchShift
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
