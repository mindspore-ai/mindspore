
.. py:function:: mindspore.tools.sdc_detect_start()

    开启静默数据错误检测。开启后，会对当前设备正反向计算中MatMul的输入输出进行校验，因此执行时间会增加，校验耗时占比随矩阵形状增大而减小。
    对于单个4096大小的MatMul计算，开启后性能下降约100%；对于Llama2-7B模型（数据并行为4，流水线并行为2，解码器使用qkv融合和ffn融合），开启后性能下降约90%。

.. py:function:: mindspore.tools.sdc_detect_stop()

    关闭静默数据错误检测。

.. py:function:: mindspore.tools.get_sdc_detect_result()

    获取静默数据错误检测结果。

    返回：
        bool，表示在检测开启后是否发生静默数据错误。
