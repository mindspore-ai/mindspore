mindspore_lite.ModelGroupFlag
=============================

.. py:class:: mindspore_lite.ModelGroupFlag

    `ModelGroupFlag` 类用于构造 `ModelGroup` 的标签。目前支持以下场景：

    1. `ModelGroupFlag.SHARE_WEIGHT` ，共享工作空间内存，`ModelGroup` 的默认构造标签。

    2. `ModelGroupFlag.SHARE_WORKSPACE` ，共享权重内存，多个模型共享权重（包括常量和变量）内存。

    3. `ModelGroupFlag.SHARE_WEIGHT_WORKSPACE` ，共享权重内存和工作空间内存。
