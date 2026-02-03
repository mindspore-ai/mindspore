mindspore.numpy.tensordot
=========================

.. py:function:: mindspore.numpy.tensordot(a, b, axes=2)

    沿指定轴计算Tensor的点积。

    给定两个Tensor `a` 和 `b` ，以及一个包含两个类数组对象的类数组对象 (a_axes, b_axes)，对 `a` 和 `b` 的元素(分量)在 `a_axes` 和 `b_axes` 指定的轴上求逐元素乘积之和。第三个参数可以是一个非负整数标量 `N` ，如果是 `N` ，则对 `a` 的最后 `N` 个维度和 `b` 的前 `N` 个维度进行求和。 常见的三种用例是：

    - ``axes = 0`` ：Tensor积
    - ``axes = 1`` ：Tensor点积
    - ``axes = 2`` ：(默认)Tensor双重缩并

    当 `axes` 为整数时，计算的顺序是：首先是 `a` 的第-N轴和 `b` 的第0轴，最后是 `a` 的 -1 轴和 `b` 的第N轴。
    对于多个轴进行求和且它们不是 `a(b)` 的最后(b的第一个)轴时，参数 `axes` 应由两个相同长度的序列组成，第一个序列应用于 `a` ，第二个序列应用于 `b` ，依此类推。
    结果的shape由第一个Tensor的未缩并轴以及第二个Tensor的未缩并轴组成。

    .. note::
        在 CPU 上，支持的dtype是 np.float16 和 np.float32。在 GPU 上，支持的dtype是 np.float16 和 np.float32。

    参数：
        - **a** (Tensor) - 需计算点积的Tensor。
        - **b** (Tensor) - 需计算点积的Tensor。
        - **axes** (int或int的序列) - 默认值： ``2`` 。

          类int值：如果是整数 `N` ，则按顺序对 `a` 的最后 `N` 个轴和 `b` 的前 `N` 个轴进行求和。对应的轴的大小必须匹配。

          int的序列：一个包含要进行求和的轴的列表，第一个序列应用于 `a` ，第二个序列应用于 `b` 。 两个类数组元素必须具有相同长度。

    返回：
        Tensor或元素为Tensor的list，输入Tensor的点积。