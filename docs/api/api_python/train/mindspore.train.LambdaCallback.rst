mindspore.train.LambdaCallback
==============================

.. py:class:: mindspore.train.LambdaCallback(on_train_epoch_begin=None, on_train_epoch_end=None, on_train_step_begin=None, on_train_step_end=None, on_train_begin=None, on_train_end=None, on_eval_epoch_begin=None, on_eval_epoch_end=None, on_eval_step_begin=None, on_eval_step_end=None, on_eval_begin=None, on_eval_end=None)

    用于自定义简单的callback。

    使用匿名函数构建callback，定义的匿名函数将在 `mindspore.train.Model.{train | eval | fit}` 的对应阶段被调用。

    请注意，callback的每个阶段都需要一个位置参数：`run_context`。

    .. note::
        - 这是一个会变更或删除的实验性接口。

    参数：
        - **on_train_epoch_begin** (Function) - 训练每个epoch开始时被调用。
        - **on_train_epoch_end** (Function) - 训练每个epoch结束时被调用。
        - **on_train_step_begin** (Function) - 训练每个step开始时被调用。
        - **on_train_step_end** (Function) - 训练每个step结束时被调用。
        - **on_train_begin** (Function) - 模型训练开始时被调用。
        - **on_train_end** (Function) - 模型训练结束时被调用。
        - **on_eval_epoch_begin** (Function) - 推理的epoch开始时被调用。
        - **on_eval_epoch_end** (Function) - 推理的epoch结束时被调用。
        - **on_eval_step_begin** (Function) - 推理的每个step开始时被调用。
        - **on_eval_step_end** (Function) - 推理的每个step结束时被调用。
        - **on_eval_begin** (Function) - 模型推理开始时被调用。
        - **on_eval_end** (Function) - 模型推理结束时被调用。
