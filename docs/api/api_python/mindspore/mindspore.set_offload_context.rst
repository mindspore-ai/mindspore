mindspore.set_offload_context
====================================

.. py:function:: mindspore.set_offload_context(offload_config)

    配置异构训练详细参数，来调整offload策略。

    .. note::
        offload配置只有在通过mindspore.set_context(memory_offload="ON")开启offload功能才会被使用，并且memory_optimize_level必须设置为O0。在Ascend硬件平台上，图编译等级必须为O0。

    参数：
        - **offload_config** (dict) - 输入格式为{"offload_path": "./offload","offload_cpu_size":"512GB","hbm_ratio":0.9}。支持以下参数配置：

          - offload_path：offload到磁盘上的路径，支持相对路径，默认值： ``"./offload"``。
          - offload_cpu_size：设置可用于offload到host侧总内存大小，格式只支持"xxGB"形式字符串。
          - offload_disk_size：设置可用于offload到磁盘的大小，格式只支持"xxGB"形式字符串。
          - hbm_ratio：策略相关参数，策略在用户设置的最大显存基础上能够使用的比例，小数，值范围(0,1]，默认值： ``1.0`` 。
          - cpu_ratio：策略相关参数，策略在用户设置的最大host侧内存基础上能够使用的比例，小数，值范围(0,1]，默认值： ``1.0`` 。
          - enable_pinned_mem：是否开启Pinned Memory，开启后可加速HBM-DDR之间的拷贝，会影响系统虚拟内存。Bool类型。默认值： ``True``。
          - enable_aio：是否开启AIO，开启后可加速DDR-NVME之间的拷贝。Bool类型。默认值： ``True`` 。
          - aio_block_size：AIO中blocksize值，格式只支持"xxGB"形式字符串。
          - aio_queue_size：AIO中depth值，刷新磁盘等待队列最大值, 取值需为整数。
          - offload_param：参数初始位置，进行设置时只支持"disk"或"cpu"，用户不设置默认为 ``""`` 。
          - offload_checkpoint：重计算点offload位置，只有开启重计算才有效，进行设置时只支持"disk"或"cpu"，用户不设置默认为 ``""`` 。
          - auto_offload：是否自动进行offload策略生成， ``True`` 时生成自动策略， ``False`` 时配合offload_param生成强制offload策略，一般取值为 ``True`` ，默认值： ``True`` 。
          - host_mem_block_size：host侧内存池block块的大小, 格式只支持"xxGB"形式字符串。通过调整块大小，可以减少内存碎片的产生。

    异常：
        - **ValueError** - 输入key不是offload_config中的属性。
