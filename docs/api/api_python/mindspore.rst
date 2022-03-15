mindspore
=========

张量
------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.Tensor
    mindspore.COOTensor
    mindspore.CSRTensor
    mindspore.RowTensor
    mindspore.SparseTensor

参数
---------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.Parameter
    mindspore.ParameterTuple

数据类型
--------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.dtype
    mindspore.dtype_to_nptype
    mindspore.issubclass_
    mindspore.dtype_to_pytype
    mindspore.pytype_to_dtype
    mindspore.get_py_obj_dtype

随机种子
---------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.set_seed
    mindspore.get_seed

模型
-----

.. mscnautosummary::
    :toctree: mindspore

    mindspore.Model

数据处理工具
-------------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.DatasetHelper
    mindspore.connect_network_with_dataset

混合精度管理
--------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.LossScaleManager
    mindspore.FixedLossScaleManager
    mindspore.DynamicLossScaleManager

序列化
-------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.save_checkpoint
    mindspore.load_checkpoint
    mindspore.load_param_into_net
    mindspore.export
    mindspore.load
    mindspore.parse_print
    mindspore.build_searched_strategy
    mindspore.merge_sliced_parameter
    mindspore.load_distributed_checkpoint
    mindspore.async_ckpt_thread_status
    mindspore.restore_group_info_list

即时编译
--------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.ms_function
    mindspore.ms_class

日志
----

.. mscnautosummary::
    :toctree: mindspore

    mindspore.get_level
    mindspore.get_log_config

自动混合精度
------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.build_train_network

安装验证
--------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.run_check

调试
------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.set_dump

内存回收
----------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.ms_memory_recycle
