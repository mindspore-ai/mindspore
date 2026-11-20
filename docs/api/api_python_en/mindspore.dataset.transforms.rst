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
    :template: classtemplate_dataset.rst

    mindspore.dataset.transforms.TypeCast

Utilities
^^^^^^^^^^

.. autosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate_dataset.rst


Vision
-------

.. automodule:: mindspore.dataset.vision

Example Gallery
^^^^^^^^^^^^^^^^

Example gallery of using vision transform APIs, jump to `Illustration of vision transforms <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/vision_gallery.html>`_.
This guide presents various transforms and input/output results.

Transforms
^^^^^^^^^^^

.. autosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.vision.CenterCrop
    mindspore.dataset.vision.Decode
    mindspore.dataset.vision.HWC2CHW
    mindspore.dataset.vision.Normalize
    mindspore.dataset.vision.RandomColorAdjust
    mindspore.dataset.vision.RandomCrop
    mindspore.dataset.vision.RandomCropDecodeResize
    mindspore.dataset.vision.RandomHorizontalFlip
    mindspore.dataset.vision.RandomResizedCrop
    mindspore.dataset.vision.RandomVerticalFlip
    mindspore.dataset.vision.Rescale
    mindspore.dataset.vision.Resize
    mindspore.dataset.vision.ToNumpy
    mindspore.dataset.vision.ToPIL
    mindspore.dataset.vision.ToTensor

Utilities
^^^^^^^^^^

.. autosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.vision.ImageReadMode
    mindspore.dataset.vision.Border
    mindspore.dataset.vision.ImageBatchFormat
    mindspore.dataset.vision.Inter

Text
-----

.. automodule:: mindspore.dataset.text

Example Gallery
^^^^^^^^^^^^^^^^

Example gallery of using vision transform APIs, jump to `Illustration of text transforms <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/text_gallery.html>`_.
This guide presents various transforms and input/output results.

Transforms
^^^^^^^^^^^

.. msnoteautosummary::
    :toctree: dataset_text
    :nosignatures:
    :template: classtemplate_dataset.rst

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
    :template: classtemplate_dataset.rst

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

Example Gallery
^^^^^^^^^^^^^^^^

Example gallery of using vision transform APIs, jump to `Illustration of audio transforms <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/audio_gallery.html>`_.
This guide presents various transforms and input/output results.

Utilities
^^^^^^^^^^

.. autosummary::
    :toctree: dataset_audio
    :nosignatures:
    :template: classtemplate_dataset.rst

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
