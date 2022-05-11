mindspore.reset_algo_parameters
================================

.. py:function:: mindspore.reset_algo_parameters()

    重置算法参数属性。

    .. note::
        此接口仅在AUTO_PARALLEL模式下工作。

    重置后，属性值为：

    - fully_use_devices：True
    - elementwise_op_strategy_follow：False
    - enable_algo_approxi：False
    - algo_approxi_epsilon：0.1
    - tensor_slice_align_enable：False
    - tensor_slice_align_size：16
