mindspore.device_context.ascend.op_precision.precision_mode
===========================================================

.. py:function:: mindspore.device_context.ascend.op_precision.precision_mode(mode)

    配置混合精度模式。框架默认设置Atlas训练系列产品的默认配置为“allow_fp32_to_fp16”，Atlas A2训练系列产品等其他产品的默认配置为“must_keep_origin_dtype”。
    如果您想了解更多详细信息，
    请查询 `昇腾社区 <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/appdevgapi/aclcppdevg_03_1371.html/>`_ 。

    .. note::
        该参数的默认值属于实验性质参数，将来可能会发生变化。

    参数：
        - **mode** (str) - 混合精度模式设置。

          其值范围如下：

          - force_fp16: 当算子既支持float16，又支持float32时，直接选择float16。
          - allow_fp32_to_fp16: 对于矩阵类算子，使用float16。对于矢量类算子，优先保持原图精度：如果网络模型中算子支持float32，则保留原始精度float32；如果网络模型中算子不支持float32，则直接降低精度到float16。
          - allow_mix_precision: 自动混合精度，针对全网算子，按照内置的优化策略，自动将部分算子的精度降低到float16或bfloat16。
          - must_keep_origin_dtype: 保持原图精度。
          - force_fp32: 当矩阵计算的算子输入为float16，输出既支持float16又支持float32时，强制转换成float32输出。
          - allow_fp32_to_bf16: 对于矩阵类算子，使用bfloat16。对于矢量类算子，优先保持原图精度：如果网络模型中算子支持float32，则保留原始精度float32；如果网络模型中算子不支持float32，则直接降低精度到bfloat16。
          - allow_mix_precision_fp16: 自动混合精度，针对全网算子，按照内置的优化策略，自动将部分算子的精度降低到float16。
          - allow_mix_precision_bf16: 自动混合精度，针对全网算子，按照内置的优化策略，自动将部分算子的精度降低到bfloat16。