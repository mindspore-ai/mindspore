mindspore.device_context.gpu.op_tuning.conv_wgrad_algo
=========================================================

.. py:function:: mindspore.device_context.gpu.op_tuning.conv_wgrad_algo(mode)

    指定cuDNN的卷积输入卷积核的反向算法。
    详细信息请查看 `NVIDIA cuDNN关于cudnnConvolutionBackwardFilter的说明 <https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html>`_。

    参数：
        - **mode** (str) - cuDNN的卷积输入卷积核的反向算法。如未配置，框架默认为 ``'normal'``。
          其值范围如下：

          - normal：使用cuDNN自带的启发式搜索算法，会根据卷积形状和类型快速选择合适的卷积算法。该参数不保证性能最优。
          - performance：使用cuDNN自带的试运行搜索算法，会根据卷积形状和类型试运行所有卷积算法，然后选择最优算法。该参数保证性能最优。
          - algo_0：该算法将卷积表示为矩阵乘积的和，而没有实际显式地形成保存输入张量数据的矩阵。求和使用原子加法操作完成，因此结果是不确定的。
          - algo_1：该算法将卷积表示为矩阵乘积，而没有实际显式地形成保存输入张量数据的矩阵。结果是确定的。
          - algo_3：该算法类似于 `algo_0`，但使用一些小的工作空间来预计算一些索引。结果也是不确定的。
          - fft：该算法利用快速傅里叶变换完成卷积计算。需要额外申请较大的内存空间，保存中间结果。结果是确定的。
          - fft_tiling：该算法利用快速傅里叶变换完成卷积计算，但是需要对输入进行分块。同样需要额外申请内存空间，保存中间结果，但是对大尺寸的输入，所需内存空间小于 fft 算法。结果是确定的。
          - winograd_nonfused：该算法利用Winograd变形算法完成卷积计算。需要额外申请较大的内存空间，保存中间结果。结果是确定的。