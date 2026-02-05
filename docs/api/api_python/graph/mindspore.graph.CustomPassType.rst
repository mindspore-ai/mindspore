mindspore.graph.CustomPassType
==============================

.. py:class:: mindspore.graph.CustomPassType

    自定义pass类型。
    可能的枚举值包括： ``CustomPassType.PATTERN`` 、 ``CustomPassType.FULL_GRAPH`` 。

    - ``CustomPassType.PATTERN``：根据用户想要匹配的图结构修改图结构，如 `Add` 算子和 `Neg` 算子替换为一个 `Sub` 算子，应用于后端优化流程。

    - ``CustomPassType.FULL_GRAPH``：基于整图信息修改图结构，应用于前端优化流程。