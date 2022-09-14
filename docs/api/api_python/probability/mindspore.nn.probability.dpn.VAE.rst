mindspore.nn.probability.dpn.VAE
================================

.. py:class:: mindspore.nn.probability.dpn.VAE(encoder, decoder, hidden_size, latent_size)

    变分自动编码器 (VAE)。
    VAE 定义了一个生成模型，Z从先验中采样，然后由解码器用于重建X。有关更多详细信息，请参阅 `自动编码变分贝叶斯 <https://arxiv.org/abs/1312.6114>`_。

    .. note:: 定义编码器和解码器时，编码器的输出 Tensor 和解码器的输入Tensor 的 shape 必须是 :math:`(N, hidden\_size)`。潜在大小必须小于或等于隐藏大小。

    参数：
        - **encoder** (Cell) - 定义为编码器的深度神经网络 (DNN) 模型。
        - **decoder** (Cell) - 定义为解码器的深度神经网络 (DNN) 模型。
        - **hidden_size** (int) - 编码器输出 Tensor 的隐藏大小。
        - **latent_size** (int) - 潜在空间的大小。

    输入：
        - **input** (Tensor) - 输入 Tensor 的 shape 是 :math:`(N, C, H, W)`，与编码器的输入相同。

    输出：
        - **output** (Tuple) - （recon_x（Tensor），x（Tensor），mu（Tensor），std（Tensor））。

    .. py:method:: generate_sample(generate_nums, shape)

        从潜在空间中随机采样以生成样本。

        参数：
            - **generate_nums** (int) - 要生成的样本数。
            - **shape** (tuple) - 样本的 shape，它必须是 :math:`(generate\_nums, C, H, W)` 或 :math:`(-1, C, H, W)`。

        返回：
            Tensor，生成的样本。

    .. py:method:: reconstruct_sample(x)

        从原始数据重建样本。

        参数：
            - **x** (Tensor) - 要重构的输入 Tensor，shape 为 :math:`(N, C, H, W)`。

        返回：
            Tensor，重构的样本。
