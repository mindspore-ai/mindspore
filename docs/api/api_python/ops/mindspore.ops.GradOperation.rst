mindspore.ops.GradOperation
============================

.. py:class:: mindspore.ops.GradOperation(get_all=False, get_by_list=False, sens_param=False)

    一个高阶函数，为输入函数生成梯度函数。

    由 `GradOperation` 高阶函数生成的梯度函数可以通过构造参数自定义。

    构建一个以x和y为输入的函数 `net = Net()` ，并带有一个参数z，详见样例中的 `Net` 。

    生成一个梯度函数，该函数返回第一个输入的梯度（见样例中的 `GradNetWrtX` ）。

    1. 构建一个带有默认参数的 `GradOperation` 高阶函数： `grad_op = GradOperation()` 。

    2. 将 `net` 作为参数调用 `grad_op` ，得到梯度函数： `gradient_function = grad_op(net)` 。

    3. 用 `net` 的输入作为参数调用梯度函数，得到第一个输入的梯度：`grad_op(net)(x, y)` 。

    生成一个梯度函数，该函数返回所有输入的梯度（见样例中的 `GradNetWrtXY` ）。

    1. 构造一个带有 `get_all=True` 参数的 `GradOperation` 高阶函数，表示获得在样例中 `Net()` 中的x和y所有输入的梯度：`grad_op = GradOperation(get_all=True)` 。
    
    2. 将 `net` 作为参数调用 `grad_op` ，得到梯度函数： `gradient_function = grad_op(net)` 。
    
    3. 用 `net` 的输入作为参数调用梯度函数，得到所有输入的梯度：`gradient_function(x, y)` 。

    生成一个梯度函数，该函数返回给定参数的梯度（见样例中的 `GradNetWithWrtParams` ）。

    1. 构造一个带有 `get_by_list=True` 参数的GradOperation高阶函数： grad_op = GradOperation(get_by_list=True)。

    2. 当构建 `GradOperation` 高阶函数时，创建一个 `ParameterTuple` 和 `net` 作为参数输入， `ParameterTuple` 作为参数过滤器决定返回哪个梯度：`params = ParameterTuple(net.trainable_params())` 。

    3. 将 `net` 和 `params` 作为参数输入 `grad_op` ，得到梯度函数： `gradient_function = grad_op(net, params)` 。

    4. 用 `net` 的输入作为参数调用梯度函数，得到关于给定参数的梯度： `gradient_function(x, y)` 。

    生成一个梯度函数，该函数以((dx, dy), (dz))的格式返回关于所有输入和给定参数的梯度（见样例中的 `GradNetWrtInputsAndParams` ）。

    1. 构建一个带有 `get_all=True` 和 `get_by_list=True` 参数的 `GradOperation` 高阶函数：`grad_op = GradOperation(get_all=True, get_by_list=True)` 。

    2. 当构建 `GradOperation` 高阶函数时，创建一个 `ParameterTuple` 和 `net` 作为参数输入：`params = ParameterTuple(net.trainable_params())` 。

    3. 将 `net` 和 `params` 作为参数输入 `grad_op` ，得到梯度函数： `gradient_function = grad_op(net, params)` 。

    4. 用 `net` 的输入作为参数调用梯度函数，得到关于所有输入和给定参数的梯度：`gradient_function(x, y)` 。

    注意：对于上面产生的梯度函数，其返回值会因返回梯度的数量不同而出现差异：
    1. 如果仅有一个梯度结果，返回单独值。
    2. 如果有多个梯度结果，返回tuple。
    3. 如果没有任何梯度结果，返回空tuple。

    我们可以设置 `sens_param` 等于True来配置灵敏度（关于输出的梯度），向梯度函数传递一个额外的灵敏度输入值。这个输入值必须与 `net` 的输出具有相同的形状和类型（见样例中的 `GradNetWrtXYWithSensParam` ）。

    1. 构建一个带有 `get_all=True` 和 `sens_param=True` 参数的 `GradOperation` 高阶函数：`grad_op = GradOperation(get_all=True, sens_param=True)` 。

    2. 当 `sens_param=True` ，定义 `grad_wrt_output` （关于输出的梯度）：`grad_wrt_output = Tensor(np.ones([2, 2]).astype(np.float32))` 。

    3. 用 `net` 作为参数输入 `grad_op` ，得到梯度函数：`gradient_function = grad_op(net)` 。

    4. 用 `net` 的输入和 `sens_param` 作为参数调用梯度函数，得到关于所有输入的梯度：`gradient_function(x, y, grad_wrt_output)` 。

    参数：
        - **get_all** (bool) - 计算梯度，如果等于False，获得第一个输入的梯度，如果等于True，获得所有输入的梯度。默认值：False。
        - **get_by_list** (bool) - 如果 `get_all` 和 `get_by_list` 都等于False，则得到第一个输入的梯度。如果等于True，获得所有Parameter自由变量的梯度。如果 `get_all` 和 `get_by_list` 都等于True，则同时得到输入和Parameter自由变量的梯度，输出形式为(“所有输入的梯度”，“所有Parameter自由变量的梯度”)。默认值：False。
        - **sens_param** (bool) - 是否在输入中配置灵敏度（关于输出的梯度）。如果sens_param等于False，自动添加一个 `ones_like(output)` 灵敏度。如果sensor_param等于True，灵敏度（输出的梯度），必须通过location参数或key-value pair参数来传递，如果是通过key-value pair参数传递value，那么key必须为sens。默认值：False。

    返回：
        将一个函数作为参数，并返回梯度函数的高阶函数。

    异常：
        - **TypeError** - 如果 `get_all` 、`get_by_list` 或者 `sens_params` 不是bool。
