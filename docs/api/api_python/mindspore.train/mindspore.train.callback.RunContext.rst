.. py:class:: mindspore.train.callback.RunContext(original_args)

    提供模型的相关信息。

    在Model方法里提供模型的相关信息。
    回调函数可以通过调用 `request_stop()` 方法来停止循环。

    **参数：**

    - **original_args** (dict) - 模型的相关信息。

    .. py:method:: get_stop_requested()

        获取是否停止训练标志。

        **返回：**

        bool，如果为True，则 `Model.train()` 停止迭代。

    .. py:method:: original_args()

        获取模型的相关信息。

        **返回：**

        dict，模型的相关信息。

    .. py:method:: request_stop()

        在训练期间设置停止请求。

        可以使用此函数请求停止训练。 `Model.train()` 会检查是否调用此函数。
