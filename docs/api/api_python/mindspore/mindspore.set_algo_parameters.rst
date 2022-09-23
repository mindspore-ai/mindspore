mindspore.set_algo_parameters
=============================

.. py:function:: mindspore.set_algo_parameters(**kwargs)

    设置并行策略搜索算法中的参数。有关典型用法，请参见 `test_auto_parallel_resnet.py <https://gitee.com/mindspore/mindspore/blob/r1.9/tests/ut/python/parallel/test_auto_parallel_resnet.py>`_ 。

    .. note::
        属性名称为必填项。此接口仅在AUTO_PARALLEL模式下工作。

    参数：
        - **fully_use_devices** (bool) - 表示是否仅搜索充分利用所有可用设备的策略。默认值：True。例如，如果有8个可用设备，当该参数设为true时，策略(4, 1)将不包括在ReLU的候选策略中，因为策略(4, 1)仅使用4个设备。
        - **elementwise_op_strategy_follow** (bool) - 表示elementwise算子是否具有与后续算子一样的策略。默认值：False。例如，Add的输出给了ReLU，其中ReLU是elementwise算子。如果该参数设置为true，则算法搜索的策略可以保证这两个算子的策略是一致的，例如，ReLU的策略(8, 1)和Add的策略((8, 1), (8, 1))。
        - **enable_algo_approxi** (bool) - 表示是否在算法中启用近似。默认值：False。由于大型DNN模型的并行搜索策略有较大的解空间，该算法在这种情况下耗时较长。为了缓解这种情况，如果该参数设置为true，则会进行近似丢弃一些候选策略，以便缩小解空间。
        - **algo_approxi_epsilon** (float) - 表示近似算法中使用的epsilon值。默认值：0.1 此值描述了近似程度。例如，一个算子的候选策略数量为S，如果 `enable_algo_approxi` 为true，则剩余策略的大小为min{S, 1/epsilon}。
        - **tensor_slice_align_enable** (bool) - 表示是否检查MatMul的tensor切片的shape。默认值：False 受某些硬件的属性限制，只有shape较大的MatMul内核才能显示出优势。如果该参数为true，则检查MatMul的切片shape以阻断不规则的shape。
        - **tensor_slice_align_size** (int) - 表示MatMul的最小tensor切片的shape，该值必须在[1,1024]范围内。默认值：16。如果 `tensor_slice_align_enable` 设为true，则MatMul tensor的最后维度的切片大小应该是该值的倍数。

    异常：
        - **ValueError** - 无法识别传入的关键字。
