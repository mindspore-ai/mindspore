mindspore.dataset.config.set_debug_mode
========================================

.. py:function:: mindspore.dataset.config.set_debug_mode(debug_mode_flag, debug_hook_list=None)

    设置是否启动数据集管道的调试模式。当启用该模式时，数据集管道会使用单线程工作，所有数据集操作均以同步方式执行。

    .. note::
        当启用调试模式时，

        - 如果随机种子没有被设置，则会将随机种子设置为1，以便在调试模式下执行数据集管道，从而获得确定性的结果。

        - 以下设置将不会生效：

          - auto_offload (强制设置为False)
          - enable_autotune (强制设置为False)
          - error_samples_mode (强制设置为ErrorSamplesMode.RETURN)
          - num_parallel_workers (强制设置为1)

        - 数据集管道中 `map` 操作的 `offload` 参数将被忽略。
        - 数据集管道中 `GeneratorDataset` 、 `map` 操作和 `batch` 操作的 `python_multiprocessing` 参数将被忽略。
        - 数据集加载API中的 `cache` 参数将会被忽略。

    参数：
        - **debug_mode_flag** (bool) - 是否开启数据集管道调试模式。该模式会强制数据集管道以单线程同步的方式运行。
        - **debug_hook_list** (list[:class:`~.dataset.debug.DebugHook`], 可选) - 调试钩子列表，用于启用调试模式时插入到 `map` 操作中各个变换操作的前后。
          默认值： ``None`` ，仅插入基础的信息打印钩子，用于打印各个变换操作输入/输出数据的形状/大小/类型信息。

    异常：
        - **TypeError** - `debug_mode_flag` 不是bool类型。
        - **TypeError** - `debug_hook_list` 不是list类型。
        - **TypeError** - `debug_hook_list` 中的元素不是 :class:`~.dataset.debug.DebugHook` 类型。
