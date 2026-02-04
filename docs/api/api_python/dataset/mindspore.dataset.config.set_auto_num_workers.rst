mindspore.dataset.config.set_auto_num_workers
===============================================

.. py:function:: mindspore.dataset.config.set_auto_num_workers(enable)

    自动为每个数据集操作设置并行线程数量（默认情况下，此功能关闭）。

    如果启用该功能，将自动调整每个数据集操作中的并行线程数量，这可能会覆盖用户通过脚本定义的并行线程数量或通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 设置的默认值（如果用户未传递任何内容）。

    目前，此函数仅针对具有per_batch_map（批处理中的运行映射）的YOLOv3数据集进行了优化。
    此功能旨在为每个操作的优化线程数量分配一个基础值。
    并行线程数有所调整的数据集操作将会被记录。

    参数：
        - **enable** (bool) - 表示是否启用自动设置线程数量的特性。

    异常：
        - **TypeError** - `enable` 不是bool类型。
