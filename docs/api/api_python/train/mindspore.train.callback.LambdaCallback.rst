.. py:class:: mindspore.train.callback.LambdaCallback(epoch_begin=None, epoch_end=None, step_begin=None, step_end=None, begin=None, end=None)

    用于自定义简单的callback。

    使用匿名函数构建callback，定义的匿名函数将在 `mindspore.Model.{train | eval}` 的对应阶段被调用。

    请注意，callback的每个阶段都需要一个位置参数：`run_context`。

    **参数：**

    - **epoch_begin** (Function) - 每个epoch开始时被调用。
    - **epoch_end** (Function) - 每个epoch结束时被调用。
    - **step_begin** (Function) - 每个step开始时被调用。
    - **step_end** (Function) - 每个step结束时被调用。
    - **begin** (Function) - 模型训练、评估开始时被调用。
    - **end** (Function) - 模型训练、评估结束时被调用。
