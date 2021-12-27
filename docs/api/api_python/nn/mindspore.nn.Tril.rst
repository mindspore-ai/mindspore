mindspore.nn.Tril
=================

.. py:class:: mindspore.nn.Tril

    返回一个Tensor，指定主对角线以上的元素被置为零。

    将矩阵元素沿主对角线分为上三角和下三角（包含对角线）。

    参数 `k` 控制对角线的选择。若 `k` 为0，则沿主对角线分割并保留下三角所有元素。若 `k` 为正值，则沿主对角线向上选择对角线 `k` ，并保留下三角所有元素。若 `k` 为负值，则沿主对角线向下选择对角线 `k` ，并保留下三角所有元素。

    **输入：**

    - **x** (Tensor)：输入Tensor。数据类型为`number <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_。
    - **k** (int)：对角线的索引。默认值：0。假设输入的矩阵的维度分别为d1，d2，则k的范围应在[-min(d1, d2)+1, min(d1, d2)-1]，超出该范围时输出值与输入 `x` 一致。

    **输出：**

    Tensor，数据类型和shape与 `x` 相同。

    **异常：**

    - **TypeError：** `k` 不是int。
    - **ValueError：** `x` 的维度小于1。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> # case1: k = 0
    >>> x = Tensor(np.array([[ 1,  2,  3,  4],
    ...                      [ 5,  6,  7,  8],
    ...                      [10, 11, 12, 13],
    ...                      [14, 15, 16, 17]]))
    >>> tril = nn.Tril()
    >>> result = tril(x)
    >>> print(result)
    [[ 1  0  0  0]
     [ 5  6  0  0]
     [10 11 12  0]
     [14 15 16 17]]
    >>> # case2: k = 1
    >>> x = Tensor(np.array([[ 1,  2,  3,  4],
    ...                      [ 5,  6,  7,  8],
    ...                      [10, 11, 12, 13],
    ...                      [14, 15, 16, 17]]))
    >>> tril = nn.Tril()
    >>> result = tril(x, 1)
    >>> print(result)
    [[ 1  2  0  0]
     [ 5  6  7  0]
     [10 11 12 13]
     [14 15 16 17]]
    >>> # case3: k = 2
    >>> x = Tensor(np.array([[ 1,  2,  3,  4],
    ...                      [ 5,  6,  7,  8],
    ...                      [10, 11, 12, 13],
    ...                      [14, 15, 16, 17]]))
    >>> tril = nn.Tril()
    >>> result = tril(x, 2)
    >>> print(result)
    [[ 1  2  3  0]
     [ 5  6  7  8]
     [10 11 12 13]
     [14 15 16 17]]
    >>> # case4: k = -1
    >>> x = Tensor(np.array([[ 1,  2,  3,  4],
    ...                      [ 5,  6,  7,  8],
    ...                      [10, 11, 12, 13],
    ...                      [14, 15, 16, 17]]))
    >>> tril = nn.Tril()
    >>> result = tril(x, -1)
    >>> print(result)
    [[ 0  0  0  0]
     [ 5  0  0  0]
     [10 11  0  0]
     [14 15 16  0]]
