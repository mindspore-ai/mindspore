.. role:: hidden
    :class: hidden-section

.. currentmodule:: {{ module }}

{% if objname in ['ArgoverseDataset', 'Caltech101Dataset', 'Caltech256Dataset', 'CelebADataset', 'Cifar100Dataset', 'Cifar10Dataset', 'CityscapesDataset', 'CocoDataset', 'DIV2KDataset', 'EMnistDataset', 'FakeImageDataset', 'FashionMnistDataset', 'FlickrDataset', 'Flowers102Dataset', 'ImageFolderDataset', 'InMemoryGraphDataset', 'KMnistDataset', 'ManifestDataset', 'MnistDataset', 'PhotoTourDataset', 'Places365Dataset', 'QMnistDataset', 'SBDataset', 'SBUDataset', 'SemeionDataset', 'STL10Dataset', 'SVHNDataset', 'USPSDataset', 'VOCDataset', 'WIDERFaceDataset']%}

{{ fullname | underline }}

.. autoclass:: {{ name }}

Pre-processing Operation
----------------------------

.. autosummary::
    :toctree: dataset_method/operation
    :nosignatures:


    mindspore.dataset.Dataset.apply
    mindspore.dataset.Dataset.concat
    mindspore.dataset.Dataset.filter
    mindspore.dataset.Dataset.flat_map
    mindspore.dataset.Dataset.map
    mindspore.dataset.Dataset.project
    mindspore.dataset.Dataset.rename
    mindspore.dataset.Dataset.repeat
    mindspore.dataset.Dataset.reset
    mindspore.dataset.Dataset.save
    mindspore.dataset.Dataset.set_dynamic_columns
    mindspore.dataset.Dataset.shuffle
    mindspore.dataset.Dataset.skip
    mindspore.dataset.Dataset.split
    mindspore.dataset.Dataset.take
    mindspore.dataset.Dataset.zip

Batch
------------------------

.. autosummary::
    :toctree: dataset_method/batch
    :nosignatures:


    mindspore.dataset.Dataset.batch
    mindspore.dataset.Dataset.bucket_batch_by_length

Iterator
---------

.. autosummary::
    :toctree: dataset_method/iterator
    :nosignatures:


    mindspore.dataset.Dataset.create_dict_iterator
    mindspore.dataset.Dataset.create_tuple_iterator

Attribute
----------

.. autosummary::
    :toctree: dataset_method/attribute
    :nosignatures:


    mindspore.dataset.Dataset.dynamic_min_max_shapes
    mindspore.dataset.Dataset.get_batch_size
    mindspore.dataset.Dataset.get_class_indexing
    mindspore.dataset.Dataset.get_col_names
    mindspore.dataset.Dataset.get_dataset_size
    mindspore.dataset.Dataset.get_repeat_count
    mindspore.dataset.Dataset.input_indexs
    mindspore.dataset.Dataset.num_classes
    mindspore.dataset.Dataset.output_shapes
    mindspore.dataset.Dataset.output_types

Apply Sampler
--------------

.. autosummary::
    :toctree: dataset_method/sampler
    :nosignatures:


    mindspore.dataset.MappableDataset.add_sampler
    mindspore.dataset.MappableDataset.use_sampler

Others
--------

.. autosummary::
    :toctree: dataset_method/others
    :nosignatures:


    mindspore.dataset.Dataset.close_pool
    mindspore.dataset.Dataset.device_que
    mindspore.dataset.Dataset.sync_update
    mindspore.dataset.Dataset.sync_wait
    mindspore.dataset.Dataset.to_json
    mindspore.dataset.Dataset.to_device


{% elif objname in ['LJSpeechDataset', 'SpeechCommandsDataset', 'TedliumDataset', 'YesNoDataset'] %}

{{ fullname | underline }}

.. autoclass:: {{ name }}

Pre-processing Operation
-------------------------

.. autosummary::
    :toctree: dataset_method/operation
    :nosignatures:


    mindspore.dataset.Dataset.apply
    mindspore.dataset.Dataset.concat
    mindspore.dataset.Dataset.filter
    mindspore.dataset.Dataset.flat_map
    mindspore.dataset.Dataset.map
    mindspore.dataset.Dataset.project
    mindspore.dataset.Dataset.rename
    mindspore.dataset.Dataset.repeat
    mindspore.dataset.Dataset.reset
    mindspore.dataset.Dataset.save
    mindspore.dataset.Dataset.set_dynamic_columns
    mindspore.dataset.Dataset.shuffle
    mindspore.dataset.Dataset.skip
    mindspore.dataset.Dataset.split
    mindspore.dataset.Dataset.take
    mindspore.dataset.Dataset.zip

Batch
------------------------

.. autosummary::
    :toctree: dataset_method/batch
    :nosignatures:


    mindspore.dataset.Dataset.batch
    mindspore.dataset.Dataset.bucket_batch_by_length

Iterator
---------

.. autosummary::
    :toctree: dataset_method/iterator
    :nosignatures:


    mindspore.dataset.Dataset.create_dict_iterator
    mindspore.dataset.Dataset.create_tuple_iterator

Attribute
----------

.. autosummary::
    :toctree: dataset_method/attribute
    :nosignatures:


    mindspore.dataset.Dataset.dynamic_min_max_shapes
    mindspore.dataset.Dataset.get_batch_size
    mindspore.dataset.Dataset.get_class_indexing
    mindspore.dataset.Dataset.get_col_names
    mindspore.dataset.Dataset.get_dataset_size
    mindspore.dataset.Dataset.get_repeat_count
    mindspore.dataset.Dataset.input_indexs
    mindspore.dataset.Dataset.num_classes
    mindspore.dataset.Dataset.output_shapes
    mindspore.dataset.Dataset.output_types

Apply Sampler
--------------

.. autosummary::
    :toctree: dataset_method/sampler
    :nosignatures:


    mindspore.dataset.MappableDataset.add_sampler
    mindspore.dataset.MappableDataset.use_sampler

Others
--------

.. autosummary::
    :toctree: dataset_method/others
    :nosignatures:


    mindspore.dataset.Dataset.close_pool
    mindspore.dataset.Dataset.device_que
    mindspore.dataset.Dataset.sync_update
    mindspore.dataset.Dataset.sync_wait
    mindspore.dataset.Dataset.to_json
    mindspore.dataset.Dataset.to_device


{% elif objname in ['AGNewsDataset', 'AmazonReviewDataset', 'CLUEDataset', 'CoNLL2000Dataset', 'CSVDataset', 'DBpediaDataset', 'EnWik9Dataset', 'GeneratorDataset', 'IMDBDataset', 'IWSLT2016Dataset', 'IWSLT2017Dataset', 'MindDataset', 'NumpySlicesDataset', 'OBSMindDataset', 'PaddedDataset', 'PennTreebankDataset', 'RandomDataset', 'SogouNewsDataset', 'TextFileDataset', 'TFRecordDataset', 'UDPOSDataset', 'WikiTextDataset', 'YahooAnswersDataset', 'YelpReviewDataset'] %}

{{ fullname | underline }}

.. autoclass:: {{ name }}

Pre-processing Operation
-------------------------

.. autosummary::
    :toctree: dataset_method/operation
    :nosignatures:


    mindspore.dataset.Dataset.apply
    mindspore.dataset.Dataset.concat
    mindspore.dataset.Dataset.filter
    mindspore.dataset.Dataset.flat_map
    mindspore.dataset.Dataset.map
    mindspore.dataset.Dataset.project
    mindspore.dataset.Dataset.rename
    mindspore.dataset.Dataset.repeat
    mindspore.dataset.Dataset.reset
    mindspore.dataset.Dataset.save
    mindspore.dataset.Dataset.set_dynamic_columns
    mindspore.dataset.Dataset.shuffle
    mindspore.dataset.Dataset.skip
    mindspore.dataset.Dataset.split
    mindspore.dataset.Dataset.take
    mindspore.dataset.Dataset.zip
    mindspore.dataset.TextBaseDataset.build_sentencepiece_vocab
    mindspore.dataset.TextBaseDataset.build_vocab

Batch
------------------------

.. autosummary::
    :toctree: dataset_method/batch
    :nosignatures:


    mindspore.dataset.Dataset.batch
    mindspore.dataset.Dataset.bucket_batch_by_length

Iterator
---------

.. autosummary::
    :toctree: dataset_method/iterator
    :nosignatures:


    mindspore.dataset.Dataset.create_dict_iterator
    mindspore.dataset.Dataset.create_tuple_iterator

Attribute
----------

.. autosummary::
    :toctree: dataset_method/attribute
    :nosignatures:


    mindspore.dataset.Dataset.dynamic_min_max_shapes
    mindspore.dataset.Dataset.get_batch_size
    mindspore.dataset.Dataset.get_class_indexing
    mindspore.dataset.Dataset.get_col_names
    mindspore.dataset.Dataset.get_dataset_size
    mindspore.dataset.Dataset.get_repeat_count
    mindspore.dataset.Dataset.input_indexs
    mindspore.dataset.Dataset.num_classes
    mindspore.dataset.Dataset.output_shapes
    mindspore.dataset.Dataset.output_types

Apply Sampler
--------------

.. autosummary::
    :toctree: dataset_method/sampler
    :nosignatures:


    mindspore.dataset.MappableDataset.add_sampler
    mindspore.dataset.MappableDataset.use_sampler

Others
--------

.. autosummary::
    :toctree: dataset_method/others
    :nosignatures:


    mindspore.dataset.Dataset.close_pool
    mindspore.dataset.Dataset.device_que
    mindspore.dataset.Dataset.sync_update
    mindspore.dataset.Dataset.sync_wait
    mindspore.dataset.Dataset.to_json
    mindspore.dataset.Dataset.to_device


{% elif fullname=="mindspore.dataset.WaitedDSCallback" %}

{{ fullname | underline }}

.. autoclass:: {{ name }}
    :members: sync_epoch_begin, sync_step_begin

{% elif objname[0].istitle() %}

{{ fullname | underline }}

.. autoclass:: {{ name }}
    :inherited-members:
    :exclude-members: parse_tree, create_ir_tree, create_runtime_obj
    :members:

{% else %}
{{ fullname | underline }}

.. autofunction:: {{ fullname }}

{% endif %}

..
  autogenerated from _templates/classtemplate.rst
  note it does not have :inherited-members: