mindspore.no_inline
=====================

.. py:function:: mindspore.no_inline(fn=None)

    指定python 函数是可复用的。该函数在前端编译为可复用的子图，该子图不inline到大图中。如果指定Cell复用，请参考lazy_inline。

    参数:
        - **fn** (function): - python 函数。 如果是Cell复用，请参考lazy_inline。

    返回:
        function，原始函数。
