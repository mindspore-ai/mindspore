mindspore.nn.probability.dpn.ConditionalVAE
===========================================

.. py:class:: mindspore.nn.probability.dpn.ConditionalVAE(encoder, decoder, hidden_size, latent_size, num_classes)

    条件变分自动编码器 (CVAE)。
    与 VAE 的区别在于 CVAE 使用标签信息。有关更多详细信息，请参阅 `基于深度条件生成模型学习结构化输出表示 <http://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models>`_。

    .. note::
        定义编码器和解码器时，编码器的输出 Tensor 和解码器的输入 Tensor 的 shape 必须是 :math:`(N, hidden\_size)`。潜在大小必须小于或等于隐藏大小。

    参数：
        - **encoder** (Cell) - 定义为编码器的深度神经网络 (DNN) 模型。 
        - **decoder** (Cell) - 定义为解码器的深度神经网络 (DNN) 模型。
        - **hidden_size** (int) - 编码器输出 Tensor 的隐藏大小。
        - **latent_size** (int) - 潜在空间的大小。
        - **num_classes** (int) - 类的数量。

    输入：
        - **input_x** (Tensor) - 输入 Tensor 的 shape 是 :math:`(N, C, H, W)`，与编码器的输入相同。 
        - **input_y** (Tensor) - 目标数据的 Tensor，shape 为 :math:`(N,)`。

    输出：
        - **output** (tuple) - tuple 包含 (recon_x(Tensor), x(Tensor), mu(Tensor), std(Tensor))。

    .. py:method:: generate_sample(sample_y, generate_nums, shape)
        
        从潜在空间中随机采样以生成样本。
    
        参数：
            - **sample_y** (Tensor) - 定义样本的标签。Tensor 的 shape (`generate_nums`, ) 和类型 mindspore.int32。 
            - **generate_nums** (int) - 要生成的样本数。
            - **shape** (tuple) - 样例的 shape，格式必须为 (`generate_nums`, C, H, W) 或 (-1, C, H, W)。

        返回：
            Tensor，生成的样本。 

    .. py:method:: reconstruct_sample(x, y)
     
        从原始数据重建样本。
        
        参数：
            - **x** (Tensor) - 重构的输入 Tensor，shape 为 (N, C, H, W)。
            - **y** (Tensor) - 输入 Tensor 的 label，shape 为 (N, C, H, W)。

        返回：
            Tensor，重建的样本。
