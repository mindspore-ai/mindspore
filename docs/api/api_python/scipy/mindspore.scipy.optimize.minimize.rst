mindspore.scipy.optimize.minimize
=================================

.. py:function:: mindspore.scipy.optimize.minimize(func, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

    最小化一个或多个变量的标量函数。

    此函数的API与SciPy匹配，但有一些细微的差异：

    当 `jac` 为None时，会使用MindSpore的自动微分功能计算 ``func`` 的反向梯度。
    ``method`` 参数是必需的。如果不指定求解器，将触发异常。
    尚未实现SciPy接口中的如下可选参数：`"hess"`、`"hessp"`、`"bounds"`、`"constraints"`、`"tol"`、`"callback"`。
    由于线搜索实现的差异，优化结果可能与SciPy不同。

    .. note::
        - `minimize` 接口当前还不支持多维Tensor输入或求微分，但有支持的计划。
        - Windows平台上还不支持 `minimize`。
        - `LAGRANGE` 方法仅在 `GPU` 上支持。

    参数：
        - **func** (Callable) - 要最小化的目标函数 :math:`fun(x,*args) -> float`，其中 `x` 是一个一维数组，其shape为 :math:`(n,)`。
          `args` 是一个Tuple，用于指定 `func` 的执行所需的所有参数。
          当 `jac` 为None时，`func` 必须能支持微分。
        - **x0** (Tensor) - 初始猜测。shape为 :math:`(n,)` 的实数数组，其中 `n` 是自变量的个数。
        - **args** (Tuple) - 传递给目标函数的额外参数。默认值：``()`` 。
        - **method** (str) - 求解器类型。应为 `“BFGS”` 和 `“LBFGS”`、`“LAGRANGE”` 中的一种。
        - **jac** (Callable, 可选) - 计算梯度向量的函数。
          只支持 `"BFGS"` 和 `"LBFGS"`。如果为None，则将使用 ``func`` 的反向梯度函数进行梯度计算。
          如果 `jac` 是可执行的，则应该是能返回梯度向量的函数：:math:`jac(x, *args) -> array\_like, shape (n,)`，其中x是一个数组，其shape为 :math:`(n,)`，`args` 是一个具有固定参数的元组。
        - **hess** (Callable, 可选) - 计算Hessian矩阵的方法。当前尚未实现。
        - **hessp** (Callable, 可选) - 目标函数的Hessian乘以任意向量 `p` 。当前尚未实现。
        - **bounds** (Sequence, 可选) - `x` 中的每个元素的 `(min, max)` 对的序列。当前尚未实现。
        - **constraints** (Callable, 可选) - 表示不等式的约束，约束中的每个函数都将 `function < 0` 表示为不等式约束。
        - **tol** (float, 可选) - 异常终止的容差范围。如需更具体的操控，请使用求解器里专门的选项。默认值：``None``。
        - **callback** (Callable, 可选) - 每次迭代后调用的可执行函数。当前尚未实现。
        - **options** (Mapping[str, Any], 可选) - 用于保存求解器可选项的字典。所有求解器方法都能支持下述通用选项。默认值：``None``。

          - ``"history_size"`` (int) - 用于更新Hession矩阵的逆的缓冲区大小，仅支持与 `method="LBFGS"` 一起使用。默认值：``20``。
          - ``"maxiter"`` (int) - 要执行的最大迭代次数。根据方法的不同，每个迭代可能会使用多个函数进行求值。

          以下选项是拉格朗日方法的专有选项：

          - ``"save_tol"`` (list) - 保存 `tol` 的列表，长度与 `constrains` 相同。
          - ``"obj_weight"`` (float) - 目标函数的权重，通常在1.0 - 100000.0之间。
          - ``"lower"`` (Tensor) - 变量的下限约束，必须具有与 `x0` 相同的shape。
          - ``"upper"`` (Tensor) - 变量的上限约束，必须具有与 `x0` 相同的shape。
          - ``"learning_rate"`` (float) - 每个Adam步骤的学习率。
          - ``"coincide_func"`` (Callable) - 子函数，表示目标函数和约束之间的公共部分，用于避免冗余计算。
          - ``"rounds"`` (int) - 更新拉格朗日乘数的次数。
          - ``"steps"`` (int) - 每执行 `steps` 次就执行一次Adam去更新拉格朗日乘数。
          - ``"log_sw"`` (bool) - 是否打印每一步的 `loss` 值。

    返回：
        优化的结果。
