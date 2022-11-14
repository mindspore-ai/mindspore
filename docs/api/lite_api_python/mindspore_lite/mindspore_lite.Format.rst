mindspore_lite.Format
=====================

.. py:class:: mindspore_lite.Format

    `Format` 类定义MindSpore Lite中Tensor的格式。

    目前，支持以下 `Format` ：

    ===========================  ===============================================
    定义                          说明
    ===========================  ===============================================
    `Format.DEFAULT`             默认格式。
    `Format.NCHW`                按批次N、通道C、高度H和宽度W的顺序存储张量数据。
    `Format.NHWC`                按批次N、高度H、宽度W和通道C的顺序存储张量数据。
    `Format.NHWC4`               C轴4字节对齐格式的 `Format.NHWC` 。
    `Format.HWKC`                按高度H、宽度W、核数K和通道C的顺序存储张量数据。
    `Format.HWCK`                按高度H、宽度W、通道C和核数K的顺序存储张量数据。
    `Format.KCHW`                按核数K、通道C、高度H和宽度W的顺序存储张量数据。
    `Format.CKHW`                按通道C、核数K、高度H和宽度W的顺序存储张量数据。
    `Format.KHWC`                按核数K、高度H、宽度W和通道C的顺序存储张量数据。
    `Format.CHWK`                按通道C、高度H、宽度W和核数K的顺序存储张量数据。
    `Format.HW`                  按高度H和宽度W的顺序存储张量数据。
    `Format.HW4`                 w轴4字节对齐格式的 `Format.HW` 。
    `Format.NC`                  按批次N和通道C的顺序存储张量数据。
    `Format.NC4`                 C轴4字节对齐格式的 `Format.NC` 。
    `Format.NC4HW4`              C轴4字节对齐和W轴4字节对齐格式的 `Format.NCHW` 。
    `Format.NCDHW`               按批次N、通道C、深度D、高度H和宽度W的顺序存储张量数据。
    `Format.NWC`                 按批次N、宽度W和通道C的顺序存储张量数据。
    `Format.NCW`                 按批次N、通道C和宽度W的顺序存储张量数据。
    `Format.NDHWC`               按批次N、深度D、高度H、宽度W和通道C的顺序存储张量数据。
    `Format.NC8HW8`              C轴8字节对齐和W轴8字节对齐格式的 `Format.NCHW` 。
    ===========================  ===============================================
