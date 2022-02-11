mindspore
=========

张量
------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.Tensor
    mindspore.RowTensor
    mindspore.SparseTensor

参数
---------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.Parameter
    mindspore.ParameterTuple

数据类型
--------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.dtype
    mindspore.dtype_to_nptype
    mindspore.dtype_to_pytype
    mindspore.get_py_obj_dtype
    mindspore.issubclass_
    mindspore.pytype_to_dtype

种子
----

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.get_seed
    mindspore.set_seed

模型
-----

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.Model

MindData数据集处理
-------------------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.DatasetHelper
    mindspore.connect_network_with_dataset

损失函数管理
------------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.DynamicLossScaleManager
    mindspore.FixedLossScaleManager
    mindspore.LossScaleManager

序列化
-------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.build_searched_strategy
    mindspore.export
    mindspore.load
    mindspore.parse_print
    mindspore.load_checkpoint
    mindspore.load_distributed_checkpoint
    mindspore.load_param_into_net
    mindspore.merge_sliced_parameter
    mindspore.save_checkpoint
    mindspore.async_ckpt_thread_status

即时编译
--------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.ms_function

日志
----

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.get_level
    mindspore.get_log_config

自动混合精度
------------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.build_train_network

安装验证
--------

.. cnmsautosummary::
    :toctree: mindspore

    mindspore.run_check
