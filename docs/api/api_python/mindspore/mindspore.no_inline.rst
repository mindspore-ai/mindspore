mindspore.no_inline
===================

.. py:function:: mindspore.no_inline(fn=None)

    指定python 函数是可复用的。该函数在前端编译为可复用的子图，该子图不inline。

    参数：
        - **fn** (function) - python 函数。如果是cell的方法，请参考 :func:`mindspore.lazy_inline`。

    返回：
        function，原始函数。
