mindspore.dataset
=================

.. automodule:: mindspore.dataset

.. note::

    The built-in ``mindspore.dataset.audio`` module and the CMUArctic, GTZAN, LibriTTS,
    LJSpeech, SpeechCommands, Tedlium and YesNo dataset readers have been removed.
    For audio input, use :class:`mindspore.dataset.GeneratorDataset` with a custom reader
    and apply custom Python transforms with :meth:`mindspore.dataset.Dataset.map`.
    This is not a drop-in replacement: verify sample rates, channel layouts, dtypes,
    split/sampling rules and numerical results for your workload.
    MindRecord, DataLoader and the general dataset pipeline remain available.

User Defined
--------------

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.GeneratorDataset

Standard Format
----------------

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.MindDataset
    mindspore.dataset.OBSMindDataset
    mindspore.dataset.TFRecordDataset

Open Source
-------------

Vision
^^^^^^^^

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.ImageFolderDataset
    mindspore.dataset.ManifestDataset

Text
^^^^^

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.CSVDataset
    mindspore.dataset.TextFileDataset

Others
-------

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.NumpySlicesDataset
    mindspore.dataset.PaddedDataset
    mindspore.dataset.RandomDataset

Sampler
--------

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_inherited_sampler.rst

    mindspore.dataset.SequentialSampler
    mindspore.dataset.RandomSampler
    mindspore.dataset.DistributedSampler

Config
-------

The config module can set or obtain the global configuration parameters of the data processing pipeline.

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.config.set_seed
    mindspore.dataset.config.get_seed
    mindspore.dataset.config.set_prefetch_size
    mindspore.dataset.config.set_num_parallel_workers
    mindspore.dataset.config.set_numa_enable
    mindspore.dataset.config.set_enable_watchdog
    mindspore.dataset.config.get_enable_watchdog
    mindspore.dataset.config.set_multiprocessing_start_method
    mindspore.dataset.config.get_multiprocessing_start_method
    mindspore.dataset.config.set_video_backend
    mindspore.dataset.config.get_video_backend

Tools
-------

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.BatchInfo
    mindspore.dataset.DatasetCache
    mindspore.dataset.DSCallback
    mindspore.dataset.Schema
    mindspore.dataset.Shuffle
    mindspore.dataset.WaitedDSCallback
