mindspore.merge_sliced_parameter
=================================

.. py:method:: mindspore.merge_sliced_parameter(sliced_parameters, strategy=None)

    将参数切片合并为一个完整的参数，用于分布式推理。关于它的细节，请参考：`保存和加载模型（HyBrid Parallel模式） <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_load_model_hybrid_parallel.html>`_。

    **参数：**

    - **sliced_parameters** (list[Parameter]) - 参数切片，按rank id进行排列。
    - **strategy** (Optional[dict]) - 参数切片策略，key为参数名称，value为该参数的切片策略。如果 `strategy` 为None，则只需按0轴顺序合并参数切片。默认值：None。

    **返回：**

    合并后的参数，包含所有数据。

    **异常：**

    - **ValueError** - 合并失败。
    - **TypeError** - `sliced_parameters` 不正确或 `strategy` 不是dict。
    - **KeyError** - 参数名称不在策略的key中。

    **样例：**

    >>> import numpy as np
    >>> from mindspore import Tensor, merge_sliced_parameter, Parameter
    >>>
    >>> sliced_parameters = [
    ...                      Parameter(Tensor(np.array([0.00023915, 0.00013939, -0.00098059])),
    ...                                "network.embedding_table"),
    ...                      Parameter(Tensor(np.array([0.00015815, 0.00015458, -0.00012125])),
    ...                                "network.embedding_table"),
    ...                      Parameter(Tensor(np.array([0.00042165, 0.00029692, -0.00007941])),
    ...                                "network.embedding_table"),
    ...                      Parameter(Tensor(np.array([0.00084451, 0.00089960, -0.00010431])),
    ...                                "network.embedding_table")]
    >>> merged_parameter = merge_sliced_parameter(sliced_parameters)
    >>> print(merged_parameter)
    Parameter (name=network.embedding_table, shape=(12,), dtype=Float64, requires_grad=True)
    