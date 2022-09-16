mindspore.dataset.audio.BorderType
==================================

.. py:class:: mindspore.dataset.audio.BorderType

    音频波形的填充模式。

    可能的枚举值包括：BorderType.CONSTANT、BorderType.EDGE、BorderType.REFLECT和BorderType.SYMMETRIC。
    
    - **BorderType.CONSTANT** - 用常量值填充边界。
    - **BorderType.EDGE** - 使用各边的边界像素值进行填充。
    - **BorderType.REFLECT** - 以各边的边界为轴进行镜像填充，忽略边界像素值。
      例如，对 [1,2,3,4] 的两侧分别填充2个元素，结果为 [3,2,1,2,3,4,3,2]。
    - **BorderType.SYMMETRIC** - 以各边的边界为轴进行对称填充，包括边界像素值。
      例如，对 [1,2,3,4] 的两侧分别填充2个元素，结果为 [2,1,1,2,3,4,4,3]。

    .. note:: 该类派生自 :class:`str` 以支持 JSON 可序列化。
