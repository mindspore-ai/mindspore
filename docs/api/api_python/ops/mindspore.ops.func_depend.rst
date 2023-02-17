mindspore.ops.depend
====================

.. py:function:: mindspore.ops.depend(value, expr)

    用来处理操作间的依赖关系。

    在大多数情况下，如果操作有作用在IO或内存上的副作用，它们将按照用户的指令依序执行。在某些情况下，如果两个操作A和B没有顺序上的依赖性，而A必须在B之前执行，我们建议使用Depend来指定它们的执行顺序。使用方法如下：

    .. code-block::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = depend(y, a)
                                --->        b = B(y)

    输入：
        - **value** (Tensor) - 应被Depend操作符返回的Tensor。
        - **expr** (Expression) - 应被执行的无输出的表达式。

    输出：
        Tensor，作为 `value` 传入的变量。