mindspore.nn.probability.toolbox.VAEAnomalyDetection
====================================================

.. py:class:: mindspore.nn.probability.toolbox.VAEAnomalyDetection(encoder, decoder, hidden_size=400, latent_size=20)

    使用 VAE 进行异常检测的工具箱。

    变分自动编码器（VAE）可用于无监督异常检测。异常分数是 sample_x 与重建 sample_x 之间的误差。如果分数高，则 X 大多是异常值。

    参数：
        - **encoder** (Cell) - 定义为编码器的深度神经网络 (DNN) 模型。
        - **decoder** (Cell) - 定义为解码器的深度神经网络 (DNN) 模型。
        - **hidden_size** (int) - 编码器输出 Tensor 的大小。默认值：400。
        - **latent_size** (int) - 潜在空间的大小。默认值：20。

    .. py:method:: predict_outlier(sample_x, threshold=100.0)

        预测样本是否为异常值。

        参数：
            - **sample_x** (Tensor) - 待预测的样本，shape 为 (N, C, H, W)。
            - **threshold** (float) - 异常值的阈值。默认值：100.0。

        返回：
            bool，样本是否为异常值。

    .. py:method:: predict_outlier_score(sample_x)

        预测异常值分数。

        参数：
            - **sample_x** (Tensor) - 待预测的样本，shape 为 (N, C, H, W)。

        返回：
            float，样本的预测异常值分数。

    .. py:method:: train(train_dataset, epochs=5)

        训练 VAE 模型。

        参数：
            - **train_dataset** (Dataset) - 用于训练模型的数据集迭代器。
            - **epochs** (int) - 数据的迭代总数。默认值：5。

        返回：               
            Cell，训练完的模型。