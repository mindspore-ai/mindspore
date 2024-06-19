mindspore.reset_auto_parallel_context
======================================

.. py:function:: mindspore.reset_auto_parallel_context()

    重置自动并行的配置为默认值。如果某个程序具有不同并行模式下的任务，需要提前调用reset_auto_parallel_context()并为下一个任务设置新的并行模式。

    - device_num：1。
    - global_rank：0。
    - gradients_mean：False。
    - gradient_fp32_sync：True。
    - parallel_mode： 'stand_alone'。
    - search_mode： 'recursive_programming'。
    - auto_parallel_search_mode： 'recursive_programming'。
    - parameter_broadcast：False。
    - strategy_ckpt_load_file： ''。
    - strategy_ckpt_save_file： ''。
    - full_batch：False。
    - enable_parallel_optimizer：False。
    - enable_alltoall：False。
    - pipeline_stages：1。
    - auto_pipeline：False。
    - fusion_threshold：64。
