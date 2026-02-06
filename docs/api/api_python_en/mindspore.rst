mindspore
=========

Data Presentation
------------------

Tensor
^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Tensor
    mindspore.tensor
    mindspore.COOTensor
    mindspore.CSRTensor
    mindspore.RowTensor
    mindspore.SparseTensor
    mindspore.is_tensor
    mindspore.from_numpy
    mindspore.frombuffer

Parameter
^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Parameter
    mindspore.ParameterTuple

DataType
^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dtype
    mindspore.QuantDtype
    mindspore.common.np_dtype

Context
--------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

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

Seed
----

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.set_seed
    mindspore.get_seed

Random Number Generator
-----------------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.get_rng_state
    mindspore.Generator
    mindspore.initial_seed
    mindspore.manual_seed
    mindspore.seed
    mindspore.set_rng_state

Serialization
-------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

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

Automatic Differentiation
---------------------------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.grad
    mindspore.get_grad
    mindspore.jacfwd
    mindspore.jacrev
    mindspore.jvp
    mindspore.register_saved_tensors_hooks
    mindspore.saved_tensors_hooks
    mindspore.value_and_grad
    mindspore.vjp

Parallel Optimization
-----------------------

Automatic Vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.vmap

Recompute
^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.recompute

JIT
---

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

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

Tool
-----

Dataset Helper
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.DatasetHelper
    mindspore.Symbol
    mindspore.connect_network_with_dataset
    mindspore.data_sink

Debugging and Tuning
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

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

Log
^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.get_level
    mindspore.get_log_config

Installation Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.run_check

Security
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.obfuscate_ckpt
    mindspore.load_obf_params_into_net
