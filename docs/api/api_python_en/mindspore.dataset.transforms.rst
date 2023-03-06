mindspore.dataset.transforms
============================

General
--------

.. automodule:: mindspore.dataset.transforms

Transforms
^^^^^^^^^^^

.. autosummary::
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

Utilities
^^^^^^^^^^

.. autosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.Relational

Vision
-------

.. automodule:: mindspore.dataset.vision

Transforms
^^^^^^^^^^^

.. autosummary::
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

Utilities
^^^^^^^^^^

.. autosummary::
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

Text
-----

.. automodule:: mindspore.dataset.text

Transforms
^^^^^^^^^^^

.. msnoteautosummary::
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


Utilities
^^^^^^^^^^

.. msnoteautosummary::
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

Audio
------

.. automodule:: mindspore.dataset.audio

Transforms
^^^^^^^^^^^

.. autosummary::
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


Utilities
^^^^^^^^^^

.. autosummary::
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
