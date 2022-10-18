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

自动微分
----------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.grad
    mindspore.value_and_grad
    mindspore.jacfwd
    mindspore.jacrev
    mindspore.jvp
    mindspore.vjp

运行环境
---------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.set_context
    mindspore.get_context
    mindspore.set_auto_parallel_context
    mindspore.get_auto_parallel_context
    mindspore.reset_auto_parallel_context
    mindspore.ParallelMode
    mindspore.set_ps_context
    mindspore.get_ps_context
    mindspore.reset_ps_context
    mindspore.set_algo_parameters
    mindspore.get_algo_parameters
    mindspore.reset_algo_parameters

并行
-------------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.shard

数据处理工具
-------------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.DatasetHelper
    mindspore.connect_network_with_dataset
    mindspore.data_sink

序列化
-------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.async_ckpt_thread_status
    mindspore.build_searched_strategy
    mindspore.convert_model
    mindspore.export
    mindspore.load
    mindspore.load_checkpoint
    mindspore.load_distributed_checkpoint
    mindspore.load_param_into_net
    mindspore.merge_pipeline_strategys
    mindspore.merge_sliced_parameter
    mindspore.parse_print
    mindspore.rank_list_for_transform
    mindspore.restore_group_info_list
    mindspore.save_checkpoint
    mindspore.transform_checkpoint_by_rank
    mindspore.transform_checkpoints

调试调优
----------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.Profiler
    mindspore.SummaryCollector
    mindspore.SummaryLandscape
    mindspore.SummaryRecord
    mindspore.set_dump

即时编译
--------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.JitConfig
    mindspore.ms_function
    mindspore.ms_class
    mindspore.mutable

日志
----

.. mscnautosummary::
    :toctree: mindspore

    mindspore.get_level
    mindspore.get_log_config


安装验证
--------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.run_check

内存回收
----------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.ms_memory_recycle

二阶优化
----------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.ConvertModelUtils
    mindspore.ConvertNetUtils
