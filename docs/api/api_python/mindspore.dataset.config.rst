mindspore.dataset.config
=========================

config模块能够设置或获取数据处理的全局配置参数。

API示例所需模块的导入代码如下：

.. code-block::

    import mindspore.dataset as ds

.. py:function:: mindspore.dataset.config.set_sending_batches(batch_num)

    在昇腾设备中使用sink_mode=True进行训练时，设置默认的发送批次。

    **参数：**

    - **batch_num** (int) - 表示总的发送批次。当设置了 `batch_num` 时，它将会等待，除非增加发送批次。默认值为0，表示将发送数据集中的所有批次。

    **异常：**

    - **TypeError** - `batch_num` 不是int类型。

.. py:function:: mindspore.dataset.config.load(file)

    从文件格式中加载项目配置。

    **参数：**

    - **file** (str) - 表示待加载的配置文件的路径。

    **异常：**

    - **RuntimeError** - 文件无效，解析失败。

.. py:function:: mindspore.dataset.config.set_seed(seed)

    如果设置了种子，生成的随机数将被固定，这有助于产生确定性结果。

    .. note::
        此函数在Python随机库和numpy.random库中设置种子，以便随机进行确定性Python增强。此函数应与创建的每个迭代器一起调用，以重置随机种子。在管道中，这并不保证 `num_parallel_workers` 大于1。

    **参数：**

    - **seed** (int) - 表示随机数量的种子。该参数用于生成确定性随机数。

    **异常：**

    - **TypeError** - `seed` 不是int类型。
    - **ValueError** - `seed` 小于0或 `seed` 大于 `UINT32_MAX(4294967295)` 时， `seed` 无效。

.. py:function:: mindspore.dataset.config.get_seed()

    获取随机数的种子。如果随机数的种子已设置，则返回设置的值，否则将返回std::mt19937::default_seed这个默认种子值。

    **返回：**

    int，表示种子的随机数量。

.. py:function:: mindspore.dataset.config.set_prefetch_size(size)

    设置管道中线程的队列容量。

    **参数：**

    - **size** (int) - 表示缓存队列的长度。

    **异常：**

    - **TypeError** - `size` 不是int类型。
    - **ValueError** - `size` 小于等于0或 `size` 大于 `INT32_MAX(2147483647)` 时，线程的队列容量无效。

    .. note::
        用于预取的总内存可能会随着工作线程数量的增加而快速增长，所以当工作线程数量大于4时，每个工作线程的预取大小将减少。
        每个工作线程在运行时预取大小将是 `prefetchsize` * (4 / `num_parallel_workers` )。

.. py:function:: mindspore.dataset.config.get_prefetch_size()

    获取数据处理管道的输出缓存队列长度。

    **返回：**

    int，表示预取的总行数。

.. py:function:: mindspore.dataset.config.set_num_parallel_workers(num)

    为并行工作线程数量设置新的全局配置默认值。
    此设置会影响所有数据集操作的并行性。

    **参数：**

    - **num** (int) - 表示并行工作线程的数量，用作为每个操作的默认值。

    **异常：**

    - **TypeError** - `num` 不是int类型。
    - **ValueError** - `num` 小于等于0或 `num` 大于 `INT32_MAX(2147483647)` 时，并行工作线程数量设置无效。

.. py:function:: mindspore.dataset.config.get_num_parallel_workers()

    获取并行工作线程数量的全局配置。
    这是并行工作线程数量的值，用于每个操作。

    **返回：**

    int，表示每个操作中默认的并行工作进程的数量。

.. py:function:: mindspore.dataset.config.set_numa_enable(numa_enable)

    设置NUMA的默认状态为启动状态。如果 `numa_enable` 为True，则需要确保安装了NUMA库。

    **参数：**

    - **numa_enable** (bool) - 表示是否使用NUMA绑定功能。

    **异常：**

    - **TypeError** - `numa_enable` 不是布尔数据类型。

.. py:function:: mindspore.dataset.config.get_numa_enable()

    获取NUMA的启动状态。
    该状态将用于所有进程。

    **返回：**

    bool，表示NUMA的启动状态。

.. py:function:: mindspore.dataset.config.set_monitor_sampling_interval(interval)

    设置监测采样的默认间隔时间（毫秒）。

    **参数：**

    - **interval** (int) - 表示用于性能监测采样的间隔时间（毫秒）。

    **异常：**

    - **TypeError** - `interval` 不是int类型。
    - **ValueError** - `interval` 小于等于0或 `interval` 大于 `INT32_MAX(2147483647)` 时， `interval` 无效。

.. py:function:: mindspore.dataset.config.get_monitor_sampling_interval()

    获取性能监控采样时间间隔的全局配置。

    **返回：**

    int，表示性能监控采样间隔时间（毫秒）。

.. py:function:: mindspore.dataset.config.set_callback_timeout(timeout)

    为DSWaitedCallback设置的默认超时时间（秒）。
    如果出现死锁，等待函数将在超时时间结束后退出。

    **参数：**

    - **timeout** (int) - 表示在出现死锁情况下，用于结束DSWaitedCallback中等待的超时时间（秒）。

    **异常：**

    - **TypeError** - `timeout` 不是int类型。
    - **ValueError** - `timeout` 小于等于0或 `timeout` 大于 `INT32_MAX(2147483647)` 时 `timeout` 无效。

.. py:function:: mindspore.dataset.config.get_callback_timeout()

    获取DSWaitedCallback的默认超时时间。
    如果出现死锁，等待的函数将在超时时间结束后退出。

    **返回：**

    int，表示在出现死锁情况下，用于结束DSWaitedCallback中的等待函数的超时时间（秒）。

.. py:function:: mindspore.dataset.config.set_auto_num_workers(enable)

    自动为每个数据集操作设置并行线程数量（默认情况下，此功能关闭）。

    如果启用该功能，将自动调整每个数据集操作中的并行线程数量，这可能会覆盖用户传入的并行线程数量或通过ds.config.set_num_parallel_workers()设置的默认值（如果用户未传递任何内容）。

    目前，此函数仅针对具有per_batch_map（batch中的运行映射）的YOLOv3数据集进行了优化。
    此功能旨在为每个操作的优化线程数量分配提供基线。
    并行线程数有所调整的数据集操作将会被记录。

    **参数：**

    - **enable** (bool) - 表示是否启用自动设置线程数量的特性。

    **异常：**

    - **TypeError** - enable不是布尔类型。

.. py:function:: mindspore.dataset.config.get_auto_num_workers()

    获取当前是否开启自动线程调整。

    **返回：**

    bool，表示是否开启自动线程调整。

.. py:function:: mindspore.dataset.config.set_enable_shared_mem(enable)

    设置共享内存标志的是否启用。如果 `shared_mem_enable` 为True，则使用共享内存队列将数据传递给为数据集操作而创建的进程，而这些数据集操作将设置 `python_multiprocessing` 为True。

    .. note::
        Windows和MacOS平台尚不支持 `set_enable_shared_mem` 。

    **参数：**

    - **enable** (bool) - 表示当 `python_multiprocessing` 为True时，是否在数据集操作中使用共享内存。

    **异常：**

    - **TypeError** - `enable` 不是布尔数据类型。

.. py:function:: mindspore.dataset.config.get_enable_shared_mem()

    获取当前是否开启共享内存。

    .. note::
        Windows和MacOS平台尚不支持 `get_enable_shared_mem` 。

    **返回：**

    bool，表示是否启用共享内存。

.. py:function:: mindspore.dataset.config.set_enable_autotune(enable, json_filepath=None)

    设置是否开启自动数据加速。

    自动数据加速用于在训练过程中根据环境资源的负载，自动调整数据处理管道全局配置，提高数据处理的速度。

    可以通过设置 `json_filepath` 将优化后的全局配置保存为JSON文件，以便后续复用。

    **参数：**

    - **enable** (bool) - 是否开启自动数据加速。
    - **json_filepath** (str，可选) - 优化后的全局配置的保存路径，当路径存在同名文件时会自动覆盖。默认值：None，表示不保存配置文件。

    **异常：**

    - **TypeError** - 当 `enable` 的类型不为布尔型。
    - **TypeError** - 当 `json_filepath` 的类型不为字符串。
    - **RuntimeError** - 当 `json_filepath` 为空字符串。
    - **RuntimeError** - 当 `json_filepath` 为目录。
    - **RuntimeError** - 当 `json_filepath` 路径不存在。
    - **RuntimeError** - 当 `json_filepath` 没有写入权限。

    .. note:: 当 `enable` 为 False 时，`json_filepath` 值将会被忽略。

.. py:function:: mindspore.dataset.config.get_enable_autotune()

    获取当前是否开启自动数据加速。

    **返回：**

    bool，表示是否开启自动数据加速。

.. py:function:: mindspore.dataset.config.set_autotune_interval(interval)

    设置自动数据加速的配置调整step间隔。

    默认设置为0，将在每个epoch结束后调整配置；否则，将每隔 `interval` 个step调整一次配置。

    **参数：**

    - **interval** (int) - 配置调整的step间隔。

    **异常：**

    - **TypeError** - 当 `interval` 类型不为整型。
    - **ValueError** - 当 `interval` 不为非负数。

.. py:function:: mindspore.dataset.config.get_autotune_interval()

    获取当前自动数据加速的配置调整step间隔。

    **返回：**

    int，自动数据加速的配置调整step间隔。

.. py:function:: mindspore.dataset.config.set_enable_watchdog(enable)

    设置watchdog Python线程是否启用。默认情况下，watchdog Python线程是启用的。watchdog Python线程负责清理卡死或假死的子进程。

    **参数：**

    - **enable** (bool) - 是否开启watchdog Python线程。默认情况下，watchdog Python线程是启用的。

    **异常：**

    - **TypeError** - `enable` 不是布尔数据类型。


.. py:function:: mindspore.dataset.config.get_enable_watchdog()

    获取当前是否开启watchdog Python线程。默认初始状态是开启。

    **返回：**

    bool，表示是否开启watchdog Python线程。

.. py:function:: mindspore.dataset.config.set_multiprocessing_timeout_interval(interval)

    设置在多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的默认时间间隔（秒）。

    **参数：**

    - **interval** (int) - 表示多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的时间间隔（秒）。

    **异常：**

    - **TypeError** - `interval` 不是int类型。
    - **ValueError** - `interval` 小于等于0或 `interval` 大于 `INT32_MAX(2147483647)` 时， `interval` 无效。

.. py:function:: mindspore.dataset.config.get_multiprocessing_timeout_interval()

    获取在多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的时间间隔的全局配置。

    **返回：**

    int，表示多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的时间间隔（秒）。

.. automodule:: mindspore.dataset.config
    :members:
