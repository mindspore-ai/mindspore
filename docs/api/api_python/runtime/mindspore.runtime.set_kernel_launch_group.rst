mindspore.runtime.set_kernel_launch_group
===========================================

.. py:function:: mindspore.runtime.set_kernel_launch_group(thread_num=2, kernel_group_num=8)

    O0模式支持算子批量并行下发接口，支持开启并行下发，并配置并行数。

    参数：
        - **thread_num** (int，可选) - 并发线程数，一般不建议增加。和现有接口 :func:`mindspore.runtime.dispatch_threads_num` 配置的线程数相互独立。默认值：``2``。
        - **kernel_group_num** (int，可选) - 算子分组总数量，每线程kernel_group_num/thread_num个组，默认值：``8``。
