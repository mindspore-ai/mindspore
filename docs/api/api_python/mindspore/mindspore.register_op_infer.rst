mindspore.graph.register_op_infer
=====================================

.. py:function:: mindspore.graph.register_op_infer(op_name, infer_name)

    为算子注册自定义推导函数。

    此装饰器允许用户为 MindSpore 中的算子注册自定义推导函数。注册的推导函数将在图编译过程中被调用，用于计算算子输出的自定义属性。推导函数的结果将存储在算子输出中，使其可被计算图中的后续操作（包括通过 `register_op_wrapper` 或 `register_op_infer` 注册的函数）访问。

    .. note::
        - 可以使用不同的 `infer_name` 值在同一算子中注册多个推导函数。
        - 推导函数将接收与其注册算子相同的参数。
        - 推导结果将在计算图中传播，可以被后续操作访问。
        - 此注册影响当前进程中指定算子的所有实例。
    
    参数：
        - **op_name** (str) - 要注册推导函数的算子名称。必须为非空字符串。
        - **infer_name** (str) - 自定义属性的名称。此名称将用作算子输出属性名称。必须为非空字符串。

    返回：
        callable，被装饰器装饰的，被注册为自定义推导函数的函数。

    异常：
        - **ValueError** - 当 `op_name` 或 `infer_name` 不是非空字符串时抛出。
        - **RuntimeError** - 当尝试为同一算子的同一 `infer_name` 注册推导函数多次时抛出。
