mindspore.ops.kaiser_window
============================

.. py:function:: mindspore.ops.kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None)

    凯瑟窗口函数。

    .. math::
        w(n) = \frac{I_{0}\left( \beta\sqrt{1 - \frac{4n^{2}}{(M - 1)^{2}}} \right)}{I_{0}(\beta)}

    n的范围为

    .. math::
        - \frac{M - 1}{2} \leq n \leq \frac{M - 1}{2}

    其中 :math:`I_0` 为零阶修正Bessel函数，:math:`M` 代表了 `window_length`。

    参数：
        - **window_length** (int) - 窗口的大小。
        - **periodic** (bool, 可选) - 如果为 ``True`` ，表示返回周期窗口。如果为 ``False`` ，表示返回对称窗口。默认 ``True`` 。
        - **beta** (float, 可选) - shape参数，当 `beta` 变大时，窗口就会变窄。默认 ``12.0`` 。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        一个一维tensor。
