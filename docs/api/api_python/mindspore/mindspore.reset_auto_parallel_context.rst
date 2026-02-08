mindspore.reset_auto_parallel_context
======================================

.. py:function:: mindspore.reset_auto_parallel_context()

    重置自动并行的配置为默认值，此接口将在后续版本中废弃，请使用接口 :class:`mindspore.parallel.auto_parallel.AutoParallel` 代替。

    - device_num: 1。
    - global_rank: 0。
    - gradients_mean: False。
    - gradient_fp32_sync: True。
    - parallel_mode: 'stand_alone'。
    - search_mode: 'recursive_programming'。
    - auto_parallel_search_mode: 'recursive_programming'。
    - parameter_broadcast: False。
    - strategy_ckpt_load_file: ''。
    - strategy_ckpt_save_file: ''。
    - full_batch: False。
    - enable_parallel_optimizer: False。
    - force_fp32_communication: False。
    - enable_alltoall: False。
    - pipeline_stages: 1。
    - pipeline_result_broadcast: False。
    - fusion_threshold: 64。
    - auto_pipeline: False。
    - dump_local_norm: False。
    - dump_local_norm_path: ''。
    - dump_device_local_norm: False。
