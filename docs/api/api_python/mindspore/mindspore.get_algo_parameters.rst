mindspore.get_algo_parameters
==============================

.. py:function:: mindspore.get_algo_parameters(attr_key)

    获取算法参数配置属性。

    .. note::
        属性名称为必填项。此接口仅在AUTO_PARALLEL模式下工作。

    参数：
        - **attr_key** (str) - 属性的key。key包括"fully_use_devices"、"elementwise_op_strategy_follow"、"enable_algo_approxi"、"algo_approxi_epsilon"、"tensor_slice_align_enable"和"tensor_slice_align_size"。对应属性的含义详见 :func:`mindspore.set_algo_parameters`。

    返回：
        根据key返回属性值。

    异常：
        - **ValueError** - 无法识别传入的关键字。
 