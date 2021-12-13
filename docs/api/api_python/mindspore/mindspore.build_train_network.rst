mindspore.build_train_network
=======================================

.. py:class:: mindspore.build_train_network(network, optimizer, loss_fn=None, level='O0', boost_level='O0', **kwargs)

    构建混合精度训练网络。

    **参数：**

    - **network** (Cell) – MindSpore的网络结构。
    - **optimizer** (Optimizer) – 优化器，用于更新参数。
    - **loss_fn** (Union[None, Cell]) – 损失函数的定义，如果为None,网络结构中应该包含损失函数。默认值：None。
    - **level** (str) – 支持["O0", "O2", "O3", "auto"]。默认值："O0"。

      - **O0** - 不进行精度变化。
      - **O2** - 使网络在float16精度下运行，如果网络结构中含有 `batchnorm` 和 `loss_fn` ，使它们在float32下运行。
      - **O3** - 使网络在float16精度下运行，并且设置 `keep_batchnorm_fp32` 为Flase。
      - **auto** - 根据不同后端设置不同的级别。在GPU上设置为O2，Ascend上设置为O3。自动设置的选项为系统推荐，在特殊场景下可能并不适用。用户可以根据网络实际情况去设置。GPU推荐O2，Ascend推荐O3， `keep_batchnorm_fp32` ， `cast_model_type` 和 `loss_scale_manager` 属性由level自动决定，有可能被 `kwargs` 参数覆盖。

    - **boost_level** (str) – `mindspore.boost` 中参数 `level` 的选项，设置boost的训练模式级别。支持["O0", "O1", "O2"]。默认值: "O0"。

      - **O0** - 不进行精度变化。
      - **O2** - 开启boost模式，性能提升20%左右，精度与原始精度相同。
      - **O3** - 开启boost模式，性能提升30%左右，准确率降低小于3%。如果设置了O1或O2模式，boost相关库将自动生效。

    - **cast_model_type** (mindspore.dtype) – 支持float16，float32。如果设置了该参数，网络将被转化为设置的数据类型，而不会根据设置的level进行转换。
    - **keep_batchnorm_fp32** (bool) – 当网络被设置为float16时，将保持Batchnorm在float32中运行。设置level不会影响该属性。
    - **loss_scale_manager** (Union[None, LossScaleManager]) – 如果为None，则不进行loss scale，否则将根据 `LossScaleManager` 进行loss scale。如果设置了， `level` 将不会影响这个属性。

    **异常：**

    - **ValueError** – 仅在GPU和Ascend上支持自动混合精度。如果设备是 CPU，则为 `ValueError`。
    - **ValueError** - 如果是CPU，则属性 `loss_scale_manager` 只能设置为 `None` 或 `FixedLossScaleManager`。