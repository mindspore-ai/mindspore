mindspore
=========

数据表达
---------

张量
^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.Tensor
    mindspore.tensor
    mindspore.COOTensor
    mindspore.CSRTensor
    mindspore.RowTensor
    mindspore.SparseTensor
    mindspore.is_tensor
    mindspore.from_numpy
    mindspore.frombuffer

参数
^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.Parameter
    mindspore.ParameterTuple

数据类型
^^^^^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.dtype
    mindspore.QuantDtype
    mindspore.common.np_dtype

运行环境
---------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.set_device
    mindspore.get_current_device
    mindspore.set_deterministic
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

随机种子
---------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.set_seed
    mindspore.get_seed

随机状态管理
--------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.get_rng_state
    mindspore.Generator
    mindspore.initial_seed
    mindspore.manual_seed
    mindspore.seed
    mindspore.set_rng_state

序列化
-------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.async_ckpt_thread_status
    mindspore.ckpt_to_safetensors
    mindspore.export
    mindspore.get_ckpt_path_with_strategy
    mindspore.load
    mindspore.load_checkpoint
    mindspore.load_checkpoint_async
    mindspore.load_mindir
    mindspore.load_param_into_net
    mindspore.onnx.export
    mindspore.safetensors_to_ckpt
    mindspore.save_checkpoint
    mindspore.save_mindir

自动微分
----------------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.grad
    mindspore.get_grad
    mindspore.jacfwd
    mindspore.jacrev
    mindspore.jvp
    mindspore.register_saved_tensors_hooks
    mindspore.saved_tensors_hooks
    mindspore.value_and_grad
    mindspore.vjp

并行优化
---------

自动向量化
^^^^^^^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.vmap

重计算
^^^^^^^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.recompute

即时编译
--------

.. mscnautosummary::
    :toctree: mindspore

    mindspore.JitConfig
    mindspore.jit
    mindspore.jit_class
    mindspore.ms_memory_recycle
    mindspore.mutable
    mindspore.constexpr
    mindspore.enable_dynamic
    mindspore.lazy_inline
    mindspore.no_inline
    mindspore.set_recursion_limit

工具
-----

数据处理工具
^^^^^^^^^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.DatasetHelper
    mindspore.Symbol
    mindspore.connect_network_with_dataset
    mindspore.data_sink

调试调优
^^^^^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.profiler.profile
    mindspore.profiler._ExperimentalConfig
    mindspore.profiler.mstx
    mindspore.profiler.DynamicProfilerMonitor
    mindspore.profiler.schedule
    mindspore.profiler.tensorboard_trace_handler
    mindspore.profiler.profiler.analyse
    mindspore.SummaryCollector
    mindspore.SummaryLandscape
    mindspore.SummaryRecord
    mindspore.set_dump
    mindspore.Profiler

日志
^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.get_level
    mindspore.get_log_config


安装验证
^^^^^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.run_check


安全
^^^^^^^^^

.. mscnautosummary::
    :toctree: mindspore

    mindspore.obfuscate_ckpt
    mindspore.load_obf_params_into_net
