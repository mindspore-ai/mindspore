mindspore.mint.broadcast_to
=============================

.. py:function:: mindspore.mint.broadcast_to(input, shape)

    将输入tensor广播到目标shape。输入维度必须小于等于目标维度，设输入shape为 :math:`(x_1, x_2, ..., x_m)`，目标shape为 :math:`(*, y_1, y_2, ..., y_m)`，其中 :math:`*` 为任意额外的维度。广播规则如下：

    依次比较 :math:`x_m` 与 :math:`y_m` 、 :math:`x_{m-1}` 与 :math:`y_{m-1}` 、...、 :math:`x_1` 与 :math:`y_1` 的值，确定是否可以广播，以及广播后输出shape对应维的值。

    - 如果相等，则这个值即为目标shape该维的值。例如输入shape为 :math:`(2, 3)` ，目标shape为 :math:`(2, 3)` ，则输出shape为 :math:`(2, 3)`。

    - 如果不相等，分以下三种情况：

      - 情况一：如果目标shape该维的值为-1，则输出shape该维的值为对应输入shape该维的值。例如输入shape为 :math:`(3, 3)` ，目标shape为 :math:`(-1, 3)` ，则输出shape为 :math:`(3, 3)` 。

      - 情况二：如果目标shape该维的值不为-1，但是输入shape该维的值为1，则输出shape该维的值为目标shape该维的值。例如输入shape为 :math:`(1, 3)` ，目标shape为 :math:`(8, 3)` ，则输出shape为 :math:`(8, 3)` 。

      - 情况三：如果两个shape对应值不满足以上情况，则说明不支持由输入shape广播到目标shape。

    至此输出shape后面m维就确定好了，现在看一下前面 :math:`*` 维，有以下两种情况：

    - 如果额外的 :math:`*` 维中不含有-1，则输入shape从低维度补充维度使之与目标shape维度一致。例如目标shape为 :math:`(3, 1, 4, 1, 5, 9)` ，输入shape为 :math:`(1, 5, 9)` ，则输入shape增维变成 :math:`(1, 1, 1, 1, 5, 9)`，根据上面提到的情况二可以得出输出shape为 :math:`(3, 1, 4, 1, 5, 9)`；

    - 如果额外的 :math:`*` 维中含有-1，说明此时该-1对应一个不存在的维度，不支持广播。例如目标shape为 :math:`(3, -1, 4, 1, 5, 9)` ，输入shape为 :math:`(1, 5, 9)` ，此时不进行增维处理，而是直接报错。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **shape** (tuple[int]) - 目标shape。

    返回：
        Tensor
