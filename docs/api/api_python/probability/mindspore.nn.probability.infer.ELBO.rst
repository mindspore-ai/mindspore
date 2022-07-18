mindspore.nn.probability.infer.ELBO
===================================

.. py:class:: mindspore.nn.probability.infer.ELBO(latent_prior="Normal", output_prior="Normal")

    Evidence Lower Bound (ELBO) 网络。
    变分推断最小化了从变分分布到后验分布的 Kullback-Leibler (KL) 散度。
    它使 ELBO 最大化，这是观测值 log p(x) 的边际概率的对数的下限。ELBO 等于负 KL 散度加上一个附加常数。更多详细信息，请参阅
    `变分推理：统计学家评论 <https://arxiv.org/abs/1601.00670>`_。

    参数：
        - **latent_prior** (str) - 潜在空间的先验分布。默认值：Normal。
          Normal：潜在空间的先验分布是正态的。
        - **output_prior** (str) - 输出数据的分布。默认值：Normal。
          Normal：如果输出数据的分布是 Normal，那么 reconstruction loss 就是 MSELoss。

    输入：
        - **input_data** (Tuple) - (recon_x(Tensor), x(Tensor), mu(Tensor), std(Tensor))。 
        - **target_data** (Tensor) - 目标 Tensor 的 shape 是 :math:`(N,)`。

    输出：
        Tensor，损失浮动张量。
