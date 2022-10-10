mindspore.reset_auto_parallel_context
======================================

.. py:function:: mindspore.reset_auto_parallel_context()

    重置自动并行的配置为默认值。

    - device_num：1。
    - global_rank：0。
    - gradients_mean：False。
    - gradient_fp32_sync：True。
    - parallel_mode：'stand_alone'。
    - search_mode：'dynamic_programming'。
    - auto_parallel_search_mode：'dynamic_programming'。
    - parameter_broadcast：False。
    - strategy_ckpt_load_file：''。
    - strategy_ckpt_save_file：''。
    - full_batch：False。
    - enable_parallel_optimizer：False。
    - enable_alltoall：False。
    - pipeline_stages：1。
    - fusion_threshold：64。
