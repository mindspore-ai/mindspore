mindspore.nn.optim_ex.Optimizer
================================

.. py:class:: mindspore.nn.optim_ex.Optimizer(learning_rate, parameters, weight_decay=0.0, loss_scale=1.0)

    用于参数更新的优化器基类。

   .. warning::
        这是一个实验性的优化器模块，需要和 `nn.lr_scheduler` 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 网络参数的列表或指定了参数组的列表。
        - **defaults** (dict)- 一个包含了优化器参数默认值的字典（当参数组未指定参数值时使用此默认值）。

    Raises:
        - **TypeError** - `learning_rate` 不是int、float、Tensor。
        - **TypeError** - `parameters` 的元素不是Parameter或字典。
        - **TypeError** - `weight_decay` 不是float或int。
        - **TypeError** - `weight_decay` 小于0。
        - **TypeError** - `learning_rate` 是一个Tensor，但是其shape大于1.

    .. py:method:: add_param_group(group_id, param_group)

        为 :class:`Optimizer.param_groups` 添加一个参数组。

        参数：
            - **group_id** (int) - group的id。
            - **param_group** (dict) - 指定了当前网络参数组的特定的优化器配置。