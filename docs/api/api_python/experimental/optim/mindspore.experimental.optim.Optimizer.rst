mindspore.experimental.optim.Optimizer
=======================================

.. py:class:: mindspore.experimental.optim.Optimizer(params, defaults)

    用于参数更新的优化器基类。

    .. warning::
        这是一个实验性的优化器基类，继续为 mint 优化器提供共享能力。experimental 下的具体优化器和 lr_scheduler 模块已移除。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **defaults** (dict) - 一个包含了优化器参数默认值的字典（当参数组未指定参数值时，使用此默认值）。

    .. py:method:: add_param_group(param_group)

        为 `Optimizer.param_groups` 属性添加一个参数组。

        参数：
            - **param_group** (dict) - 指定了当前网络参数组的特定的优化器配置。