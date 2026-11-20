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

Text processing API removal
---------------------------

``mindspore.dataset.text`` and its vocabulary, tokenizer and text transform APIs have been removed.
Dataset vocabulary-building methods and serialized pipelines containing these transforms must also be migrated.
Text dataset readers, generic ``Dataset.map``, MindRecord and DataLoader remain available.
Preprocess text into token IDs, or call a Python text-processing function of your choice through ``Dataset.map``.
Check vocabularies and special tokens, case handling and Unicode normalization, tokenization, offset units,
shape/dtype and throughput. External tokenizers are not guaranteed to be equivalent; vocabulary training
and model file compatibility require separate validation.

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
