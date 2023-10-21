mindspore.dataset.text.NormalizeForm
=====================================

.. py:class:: mindspore.dataset.text.NormalizeForm

    `Unicode规范化模式 <http://unicode.org/reports/tr15/>`_ 。

    可选值如下：

    - **NormalizeForm.NONE** - 不进行规范化处理。
    - **NormalizeForm.NFC** - 先以标准等价方式分解，再以标准等价方式重组。
    - **NormalizeForm.NFKC** - 先以兼容等价方式分解，再以标准等价方式重组。
    - **NormalizeForm.NFD** - 以标准等价方式分解。
    - **NormalizeForm.NFKD** - 以兼容等价方式分解。
