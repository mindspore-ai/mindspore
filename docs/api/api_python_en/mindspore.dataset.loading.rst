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

    mindspore.dataset.Caltech101Dataset
    mindspore.dataset.Caltech256Dataset
    mindspore.dataset.CelebADataset
    mindspore.dataset.Cifar10Dataset
    mindspore.dataset.Cifar100Dataset
    mindspore.dataset.CityscapesDataset
    mindspore.dataset.CocoDataset
    mindspore.dataset.DIV2KDataset
    mindspore.dataset.EMnistDataset
    mindspore.dataset.FakeImageDataset
    mindspore.dataset.FashionMnistDataset
    mindspore.dataset.FlickrDataset
    mindspore.dataset.Flowers102Dataset
    mindspore.dataset.Food101Dataset
    mindspore.dataset.ImageFolderDataset
    mindspore.dataset.KITTIDataset
    mindspore.dataset.KMnistDataset
    mindspore.dataset.LFWDataset
    mindspore.dataset.LSUNDataset
    mindspore.dataset.ManifestDataset
    mindspore.dataset.MnistDataset
    mindspore.dataset.OmniglotDataset
    mindspore.dataset.PhotoTourDataset
    mindspore.dataset.Places365Dataset
    mindspore.dataset.QMnistDataset
    mindspore.dataset.RenderedSST2Dataset
    mindspore.dataset.SBDataset
    mindspore.dataset.SBUDataset
    mindspore.dataset.SemeionDataset
    mindspore.dataset.STL10Dataset
    mindspore.dataset.SUN397Dataset
    mindspore.dataset.SVHNDataset
    mindspore.dataset.USPSDataset
    mindspore.dataset.VOCDataset
    mindspore.dataset.WIDERFaceDataset

Text
^^^^^

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.AGNewsDataset
    mindspore.dataset.AmazonReviewDataset
    mindspore.dataset.CLUEDataset
    mindspore.dataset.CSVDataset
    mindspore.dataset.CoNLL2000Dataset
    mindspore.dataset.DBpediaDataset
    mindspore.dataset.EnWik9Dataset
    mindspore.dataset.IMDBDataset
    mindspore.dataset.IWSLT2016Dataset
    mindspore.dataset.IWSLT2017Dataset
    mindspore.dataset.Multi30kDataset
    mindspore.dataset.PennTreebankDataset
    mindspore.dataset.SogouNewsDataset
    mindspore.dataset.SQuADDataset
    mindspore.dataset.SST2Dataset
    mindspore.dataset.TextFileDataset
    mindspore.dataset.UDPOSDataset
    mindspore.dataset.WikiTextDataset
    mindspore.dataset.YahooAnswersDataset
    mindspore.dataset.YelpReviewDataset

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

    mindspore.dataset.DistributedSampler
    mindspore.dataset.PKSampler
    mindspore.dataset.RandomSampler
    mindspore.dataset.SequentialSampler
    mindspore.dataset.SubsetRandomSampler
    mindspore.dataset.SubsetSampler
    mindspore.dataset.WeightedRandomSampler

Config
-------

The config module can set or obtain the global configuration parameters of the data processing pipeline.

.. autosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_dataset.rst

    mindspore.dataset.config.set_sending_batches
    mindspore.dataset.config.load
    mindspore.dataset.config.set_seed
    mindspore.dataset.config.get_seed
    mindspore.dataset.config.set_prefetch_size
    mindspore.dataset.config.get_prefetch_size
    mindspore.dataset.config.set_num_parallel_workers
    mindspore.dataset.config.get_num_parallel_workers
    mindspore.dataset.config.set_numa_enable
    mindspore.dataset.config.get_numa_enable
    mindspore.dataset.config.set_monitor_sampling_interval
    mindspore.dataset.config.get_monitor_sampling_interval
    mindspore.dataset.config.set_callback_timeout
    mindspore.dataset.config.get_callback_timeout
    mindspore.dataset.config.set_auto_num_workers
    mindspore.dataset.config.get_auto_num_workers
    mindspore.dataset.config.set_enable_shared_mem
    mindspore.dataset.config.get_enable_shared_mem
    mindspore.dataset.config.set_enable_autotune
    mindspore.dataset.config.get_enable_autotune
    mindspore.dataset.config.set_autotune_interval
    mindspore.dataset.config.get_autotune_interval
    mindspore.dataset.config.set_auto_offload
    mindspore.dataset.config.get_auto_offload
    mindspore.dataset.config.set_enable_watchdog
    mindspore.dataset.config.get_enable_watchdog
    mindspore.dataset.config.set_fast_recovery
    mindspore.dataset.config.get_fast_recovery
    mindspore.dataset.config.set_multiprocessing_timeout_interval
    mindspore.dataset.config.get_multiprocessing_timeout_interval
    mindspore.dataset.config.set_error_samples_mode
    mindspore.dataset.config.get_error_samples_mode
    mindspore.dataset.config.ErrorSamplesMode
    mindspore.dataset.config.set_debug_mode
    mindspore.dataset.config.get_debug_mode
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
    mindspore.dataset.compare
    mindspore.dataset.debug.DebugHook
    mindspore.dataset.deserialize
    mindspore.dataset.serialize
    mindspore.dataset.show
    mindspore.dataset.utils.imshow_det_bbox
    mindspore.dataset.utils.LineReader
