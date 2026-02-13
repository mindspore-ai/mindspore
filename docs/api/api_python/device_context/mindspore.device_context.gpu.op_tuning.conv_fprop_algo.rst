mindspore.device_context.gpu.op_tuning.conv_fprop_algo
=========================================================

.. py:function:: mindspore.device_context.gpu.op_tuning.conv_fprop_algo(mode)

    指定cuDNN的卷积前向算法。
    详细信息请查看 `NVIDIA cuDNN关于cudnnConvolutionForward的说明 <https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html>`_。

    参数：
        - **mode** (str) - cuDNN的卷积前向算法。如未配置，框架默认为 ``'normal'``。
          其值范围如下：

          - normal：使用cuDNN自带的启发式搜索算法，会根据卷积形状和类型快速选择合适的卷积算法。该参数不保证性能最优。
          - performance：使用cuDNN自带的试运行搜索算法，会根据卷积形状和类型试运行所有卷积算法，然后选择最优算法。该参数保证性能最优。
          - implicit_gemm：该算法将卷积隐式转换成矩阵乘法，完成计算。不需要显式将输入张量数据转换成矩阵形式保存。
          - precomp_gemm：该算法将卷积隐式转换成矩阵乘法，完成计算。但是需要一些额外的内存空间去保存预计算得到的索引值，以便隐式地将输入张量数据转换成矩阵形式。
          - gemm：该算法将卷积显式转换成矩阵乘法，完成计算。在显式完成矩阵乘法过程中，需要额外申请内存空间，将输入转换成矩阵形式。
          - direct：该算法直接完成卷积计算，不会隐式或显式地将卷积转换成矩阵乘法。
          - fft：该算法利用快速傅里叶变换完成卷积计算。需要额外申请内存空间，保存中间结果。
          - fft_tiling：该算法利用快速傅里叶变换完成卷积计算，但是需要对输入进行分块。同样需要额外申请内存空间，保存中间结果，但是对大尺寸的输入，所需内存空间小于fft算法。
          - winograd：该算法利用Winograd变换完成卷积计算。需要额外申请合理的内存空间，保存中间结果。
          - winograd_nonfused：该算法利用Winograd变形算法完成卷积计算。需要额外申请较大的内存空间，保存中间结果。
