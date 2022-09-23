mindspore.merge_sliced_parameter
=================================

.. py:function:: mindspore.merge_sliced_parameter(sliced_parameters, strategy=None)

    将参数切片合并为一个完整的参数，用于分布式推理。关于它的细节，请参考：`保存和加载模型（HyBrid Parallel模式） <https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/parallel/save_load.html>`_。

    参数：
        - **sliced_parameters** (list[Parameter]) - 参数切片，按rank id进行排列。
        - **strategy** (Optional[dict]) - 参数切片策略，key为参数名称，value为该参数的切片策略。如果 `strategy` 为None，则只需按0轴顺序合并参数切片。默认值：None。

    返回：
        合并后的参数，包含所有数据。

    异常：
        - **ValueError** - 合并失败。
        - **TypeError** - `sliced_parameters` 不正确或 `strategy` 不是dict。
        - **KeyError** - 参数名称不在策略的key中。
