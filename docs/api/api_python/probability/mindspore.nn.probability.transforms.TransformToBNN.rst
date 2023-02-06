mindspore.nn.probability.transforms.TransformToBNN
==================================================

.. py:class:: mindspore.nn.probability.transforms.TransformToBNN(trainable_dnn, dnn_factor=1, bnn_factor=1)

    将深度神经网络 (DNN) 模型转换为贝叶斯神经网络 (BNN) 模型。

    参数：
        - **trainable_dnn** (Cell) - 由 TrainOneStepCell 包装的可训练 DNN 模型（backbone）。
        - **dnn_factor** (int, float) - backbone 的损失系数，由损失函数计算。默认值：1。
        - **bnn_factor** (int, float) - KL 损失系数，即贝叶斯层的 KL 散度。默认值：1。

    .. py:method:: transform_to_bnn_layer(dnn_layer_type, bnn_layer_type, get_args=None, add_args=None)

        将 DNN 模型中的特定类型的层转换为相应的 BNN 层。

        参数：
            - **dnn_layer_type** (Cell) - 要转换为 BNN 层的 DNN 层的类型。可选值为 nn.Dense 和 nn.Conv2d。
            - **bnn_layer_type** (Cell) - 要转换到的 BNN 层的类型。可选值是 DenseReparam 和 ConvReparam。
            - **get_args** - 从 DNN 层获得的参数。默认值：None。
            - **add_args** (dict) - 添加到 BNN 层的新参数。请注意， `add_args` 中的参数不得与 `get_args` 中的参数重复。默认值：None。

        返回：
            Cell，由 TrainOneStepCell 包裹的可训练模型，其特定类型的层被转换为对应的贝叶斯层。

    .. py:method:: transform_to_bnn_model(get_dense_args=lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias, "out_channels": dp.out_channels, "activation": dp.activation}, get_conv_args=lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias, "padding": dp.padding, "dilation": dp.dilation, "group": dp.group}, add_dense_args=None, add_conv_args=None)

        将整个DNN模型转换为BNN模型，并通过TrainOneStepCell封装BNN模型。

        参数：
            - **get_dense_args** - 从 DNN 全连接层获得的参数。默认值：lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias, "out_channels": dp.out_channels, "activation": dp.activation}。
            - **get_conv_args** - 从 DNN 卷积层获得的参数。默认值：lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias, "padding": dp.padding, "dilation": dp.dilation, "group": dp.group}。 
            - **add_dense_args** (dict) - 添加到 BNN 全连接层的新参数。请注意， `add_dense_args` 中的参数不得与 `get_dense_args` 中的参数重复。默认值：None。
            - **add_conv_args** (dict) - 添加到 BNN 卷积层的新参数。请注意， `add_conv_args` 中的参数不得与 `get_conv_args` 中的参数重复。默认值：None。

        返回：
            Cell，由 TrainOneStepCell 封装的可训练 BNN 模型。


    