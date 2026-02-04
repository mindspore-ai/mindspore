mindspore.dataset.debug.DebugHook
==================================

.. py:class:: mindspore.dataset.debug.DebugHook(prev_op_name=None)

    数据集管道Python调试器钩子的基类。所有用户定义的钩子行为都必须继承这个基类。

    为了调试数据集管道中 `map` 操作的输入和输出数据，用户可以在该类的 `compute` 函数中添加断点，可以打印日志查看数据的类型和形状等。

    参数：
        - **prev_op_name** (str, 可选) - 上一个调试点的变换名称，一般不需指定。默认值： ``None`` 。

    .. py:method:: compute(*args)
        :abstractmethod:

        定义该调试钩子的行为，此方法需要被子类重写。参考上述样例如何自定义一个调试钩子。

        参数：
            - **args** (Any) - 被调试的变换的输入/输出数据，可以直接打印查看。
