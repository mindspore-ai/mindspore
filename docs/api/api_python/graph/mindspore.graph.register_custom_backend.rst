mindspore.graph.register_custom_backend
=======================================

.. py:function:: mindspore.graph.register_custom_backend(backend_name, backend_path)

    向MindSpore注册自定义后端，使用自定义后端进行模型编译、执行。

    .. note::
        - 该自定义后端仅在使用@jit时生效，且@jit装饰器内的backend参数必须和注册的backend_name名称一致。
        - 该接口仅支持Linux系统。

    参数：
        - **backend_name** (str) - 自定义后端名称。
        - **backend_path** (str) - 自定义后端的绝对路径，以so结尾。

    返回：
        bool。若自定义后端注册成功，则返回True；否则返回False。

    异常：
        - **ValueError** - 如果自定义后端路径不存在或者文件无效。
