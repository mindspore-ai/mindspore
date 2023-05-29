mindspore.train.ConvertModelUtils
==================================

.. py:class:: mindspore.train.ConvertModelUtils

    该接口用于增加计算图，提升二阶算法THOR运行时的性能。

    .. py:method:: convert_to_thor_model(model, network, loss_fn=None, optimizer=None, metrics=None, amp_level="O0", loss_scale_manager=None, keep_batchnorm_fp32=False)
        :staticmethod:

        该接口用于增加计算图，提升二阶算法THOR运行时的性能。

        参数：
            - **model** (Object) - 用于训练的高级API。 
            - **network** (Cell) - 训练网络。
            - **loss_fn** (Cell) - 目标函数。默认值： ``None`` 。
            - **optimizer** (Cell) - 用于更新权重的优化器。默认值： ``None`` 。
            - **metrics** (Union[dict, set]) - 在训练期间由模型评估的词典或一组度量。例如：{'accuracy', 'recall'}。默认值： ``None`` 。
            - **amp_level** (str) - 混合精度训练的级别。支持["O0", "O2", "O3", "auto"]。默认值： ``"O0"`` 。

              - **O0** - 不改变。
              - **O2** - 将网络转换为float16，使用动态loss scale保持BN在float32中运行。
              - **O3** - 将网络强制转换为float16，并使用附加属性 `keep_batchnorm_fp32=False` 。
              - **auto** - 在不同设备中，将级别设置为建议级别。GPU上建议使用O2，Ascend上建议使用O3。建议级别基于专家经验，不能总是一概而论。对于特殊网络，用户需要指定对应的混合精度训练级别。

            - **loss_scale_manager** (Union[None, LossScaleManager]) - 如果为None，则不会按比例缩放loss。否则，需设置LossScaleManager，且优化器的入参loss_scale不为None。这是一个关键参数。例如，使用 `loss_scale_manager=None` 设置值。默认值： ``None`` 。
            - **keep_batchnorm_fp32** (bool) - 保持BN在 `float32` 中运行。如果为True，则将覆盖之前的级别设置。默认值： ``False`` 。

        返回：
            model (Object)，用于训练的高级API。
