云云联邦学习
================================

.. py:class:: mindspore.FederatedLearningManager(model, sync_frequency, sync_type="fixed", **kwargs)

    在训练过程中管理联邦学习。

    参数：
        - **model** (nn.Cell) - 一个用于联邦训练的模型。
        - **sync_frequency** (int) - 联邦学习中的参数同步频率。
          需要注意在数据下沉模式中，频率的单位是epoch的数量。否则，频率的单位是step的数量。
          在自适应同步频率模式下为初始同步频率，在固定频率模式下为同步频率。
        - **sync_type** (str) - 采用同步策略类型的参数。
          支持["fixed", "adaptive"]。默认值："fixed"。

          - fixed：参数的同步频率是固定的。
          - adaptive：参数的同步频率是自适应变化的。
        - **min_consistent_rate** (float) - 最小一致性比率阈值，该值越大同步频率提升难度越大。
          取值范围：大于等于0.0。默认值：1.1。
        - **min_consistent_rate_at_round** (int) - 最小一致性比率阈值的轮数，该值越大同步频率提升难度越大。
          取值范围：大于等于0。默认值：0。
        - **ema_alpha** (float) - 梯度一致性平滑系数，该值越小越会根据当前轮次的梯度分叉情况来判断频率是否
          需要改变，反之则会更加根据历史梯度分叉情况来判断。
          取值范围：(0.0, 1.0)。默认值：0.5。
        - **observation_window_size** (int) - 观察时间窗的轮数，该值越大同步频率减小难度越大。
          取值范围：大于0。默认值：5。
        - **frequency_increase_ratio** (int) - 频率提升幅度，该值越大频率提升幅度越大。
          取值范围：大于0。默认值：2。
        - **unchanged_round** (int) - 频率不发生变化的轮数，在前unchanged_round个轮次，频率不会发生变化。
          取值范围：大于等于0。默认值：0。

    .. note::
        这是一个实验原型，可能会有变化。

    .. py:method:: step_end(run_context)

        在step结束时同步参数。如果 `sync_type` 是"adaptive"，同步频率会在这里自适应的调整。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。
