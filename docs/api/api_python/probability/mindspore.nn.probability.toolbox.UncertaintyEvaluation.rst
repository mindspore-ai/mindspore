mindspore.nn.probability.toolbox.UncertaintyEvaluation
======================================================

.. py:class:: mindspore.nn.probability.toolbox.UncertaintyEvaluation(model, train_dataset, task_type, num_classes=None, epochs=1, epi_uncer_model_path=None, ale_uncer_model_path=None, save_model=False)

    包含数据不确定性和模型不确定性的评估工具箱。

    参数：
        - **model** (Cell) - 不确定性评估的模型。
        - **train_dataset** (Dataset) - 用于训练模型的数据集迭代器。
        - **task_type** (str) - 模型任务类型的选项。
          - regression：回归模型。
          - classification：分类模型。
        - **num_classes** (int) - 分类标签的数量。如果任务类型为分类，则必须设置；否则，它是不需要的。默认值：None。
        - **epochs** (int) - 数据的迭代总数。默认值：1。
        - **epi_uncer_model_path** (str) - 认知不确定性模型的保存或读取路径。默认值：None。
        - **ale_uncer_model_path** (str) - 任意不确定性模型的保存或读取路径。默认值：None。
        - **save_model** (bool) - 是否保存不确定性模型，如果为 true，`epi_uncer_model_path` 和 `ale_uncer_model_path` 不能为 None。
          如果为 false，则从不确定性模型的路径中加载要评估的模型；如果未给出路径，则不会保存或加载不确定性模型。默认值：false。

    .. py:method:: eval_aleatoric_uncertainty(eval_data)

        评估推理结果的任意不确定性，也称为数据不确定性。

        参数：
            - **eval_data** (Tensor) - 要评估的数据样本，shape 必须是 (N,C,H,W)。

        返回：            
            numpy.dtype，数据样本推断结果的任意不确定性。

    .. py:method:: eval_epistemic_uncertainty(eval_data)

        评估推理结果的认知不确定性，也称为模型不确定性。

        参数：
            - **eval_data** (Tensor) - 要评估的数据样本，shape 必须是 (N,C,H,W)。

        返回：            
            numpy.dtype，数据样本推断结果的任意不确定性。