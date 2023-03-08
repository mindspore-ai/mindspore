mindspore.ops.broadcast_to
==========================

.. py:function:: mindspore.ops.broadcast_to(input, shape)

    将输入shape广播到目标shape。输入shape维度必须小于等于目标shape维度，设输入shape为 :math:`(x_1, x_2, ..., x_m)`，目标shape为 :math:`(*, y_1, y_2, ..., y_m)`，其中 :math:`*` 为任意额外的维度。广播规则如下：

    依次比较 :math:`x_m` 与 :math:`y_m` 、 :math:`x_{m-1}` 与 :math:`y_{m-1}` 、...、 :math:`x_1` 与 :math:`y_1` 的值确定是否可以广播以及广播后输出shape对应维的值。

    - 如果相等，则这个值即为目标shape该维的值。比如说输入shape为 :math:`(2, 3)` ，目标shape为 :math:`(2, 3)` ，则输出shape为 :math:`(2, 3)`。

    - 如果不相等，分以下三种情况：

      - 情况一：如果目标shape该维的值为-1，则输出shape该维的值为对应输入shape该维的值。比如说输入shape为 :math:`(3, 3)` ，目标shape为 :math:`(-1, 3)` ，则输出shape为 :math:`(3, 3)` ；

      - 情况二：如果目标shape该维的值不为-1，但是输入shape该维的值为1，则输出shape该维的值为目标shape该维的值。比如说输入shape为 :math:`(1, 3)` ，目标shape为 :math:`(8, 3)` ，则输出shape为 :math:`(8, 3)` ；

      - 情况三：如果两个shape对应值不满足以上情况则说明不支持由输入shape广播到目标shape。

    至此输出shape后面m维就确定好了，现在看一下前面 :math:`*` 维，有以下两种情况：

    - 如果额外的 :math:`*` 维中不含有-1，则输入shape从低维度补充维度使之与目标shape维度一致，比如说目标shape为 :math:`(3, 1, 4, 1, 5, 9)` ，输入shape为 :math:`(1, 5, 9)` ，则输入shape增维变成 :math:`(1, 1, 1, 1, 5, 9)`，根据上面提到的情况二可以得出输出shape为 :math:`(3, 1, 4, 1, 5, 9)`；

    - 如果额外的 :math:`*` 维中含有-1，说明此时该-1对应一个不存在的维度，不支持广播。比如说目标shape为 :math:`(3, -1, 4, 1, 5, 9)` ，输入shape为 :math:`(1, 5, 9)` ，此时不进行增维处理，而是直接报错。

    参数：
        - **input** (Tensor) - 第一个输入，任意维度的Tensor，数据类型为float16、float32、int32、int8、uint8、bool。
        - **shape** (tuple) - 第二个输入，指定广播到目标 `shape`。

    返回：
        Tensor，shape与目标 `shape` 相同，数据类型与 `input` 相同。

    异常：
        - **TypeError** - `shape` 不是tuple。
        - **ValueError** - 输入shape 无法广播到目标 `shape` ，或者目标 `shape` 中的-1维度位于一个无效位置。
