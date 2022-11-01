预处理操作
----------

.. mscnautosummary::
    :toctree: dataset_method/operation
    :nosignatures:
    :template: classtemplate.rst

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
    mindspore.dataset.Dataset.shuffle
    mindspore.dataset.Dataset.skip
    mindspore.dataset.Dataset.split
    mindspore.dataset.Dataset.take
    mindspore.dataset.Dataset.zip

Batch（批操作）
------------------------

.. mscnautosummary::
    :toctree: dataset_method/batch
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.Dataset.batch
    mindspore.dataset.Dataset.bucket_batch_by_length
    mindspore.dataset.Dataset.padded_batch

迭代器
------

.. mscnautosummary::
    :toctree: dataset_method/iterator
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.Dataset.create_dict_iterator
    mindspore.dataset.Dataset.create_tuple_iterator

数据集属性
----------

.. mscnautosummary::
    :toctree: dataset_method/attribute
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.Dataset.get_batch_size
    mindspore.dataset.Dataset.get_class_indexing
    mindspore.dataset.Dataset.get_col_names
    mindspore.dataset.Dataset.get_dataset_size
    mindspore.dataset.Dataset.get_repeat_count
    mindspore.dataset.Dataset.input_indexs
    mindspore.dataset.Dataset.num_classes
    mindspore.dataset.Dataset.output_shapes
    mindspore.dataset.Dataset.output_types

应用采样方法
------------

.. mscnautosummary::
    :toctree: dataset_method/sampler
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.MappableDataset.add_sampler
    mindspore.dataset.MappableDataset.use_sampler

其他方法
--------

.. mscnautosummary::
    :toctree: dataset_method/others
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.Dataset.device_que
    mindspore.dataset.Dataset.sync_update
    mindspore.dataset.Dataset.sync_wait
    mindspore.dataset.Dataset.to_json
