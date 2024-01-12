mindspore.scipy.optimize.line_search
====================================

.. py:function:: mindspore.scipy.optimize.line_search(f, xk, pk, jac=None, gfk=None, old_fval=None, old_old_fval=None, c1=0.0001, c2=0.9, maxiter=20)

    满足强Wolfe条件的非精确线搜索。

    来自Wright和Nocedal，'Numerical Optimization'，1999，第59-61页，算法3.5章节。

    .. note::
        Windows平台上还不支持 `line_search`。

    参数：
        - **f** (function) - 形式为f(x)的函数，其中x是一个扁平Tensor，并返回一个实数标量。
          该函数应该由 `vjp` 定义的算子组成。
        - **xk** (Tensor) - 初始猜测。
        - **pk** (Tensor) - 要搜索的方向。假定方向是下降方向。
        - **jac** (function) - 求x处的梯度的函数，其中x是一个扁平的Tensor，函数返回一个Tensor。
          如果要使用自动微分，则可以传 ``None``。
        - **gfk** (Tensor) - `value_and_gradient` 作为位置的初始值。默认值：``None``。
        - **old_fval** (Tensor) - 同 `gfk`。默认值：``None``。
        - **old_old_fval** (Tensor) - 未使用的参数，仅用于scipy API合规性。默认值：``None``。
        - **c1** (float) - Wolfe准则常量，参见ref。默认值：``1e-4``。
        - **c2** (float) - 与 `c1` 相同。默认值：``0.9``。
        - **maxiter** (int) - 搜索的最大迭代次数。默认值：``20``。

    返回：
        线搜索的结果。
