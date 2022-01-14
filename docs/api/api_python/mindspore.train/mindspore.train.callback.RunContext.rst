.. py:class:: mindspore.train.callback.RunContext(original_args)

    提供模型的相关信息。

    在Model方法里提供模型的相关信息。
    回调函数可以调用 `request_stop()` 方法来停止迭代。
    该类需要与:class:`mindspore.train.callback.Callback`一起使用。
    有关自定义Callback的详细信息，请查看
    `Callback <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html>`_。

    **参数：**

    - **original_args** (dict) - 模型的相关信息。

    .. py:method:: get_stop_requested()

        获取是否停止训练的标志。

        **返回：**

        bool，如果为True，则 `Model.train()` 停止迭代。

    .. py:method:: original_args()

        获取模型相关信息的对象。

        **返回：**

        dict，含有模型的相关信息的对象。

    .. py:method:: request_stop()

        在训练期间设置停止请求。

        可以使用此函数请求停止训练。 `Model.train()` 会检查是否调用此函数。
