mindspore.rewrite
=================

For a complete ReWrite example, refer to
`rewrite_example.py <https://gitee.com/mindspore/mindspore/tree/r2.0/docs/api/api_python_en/rewrite_example.py>`_ 。
The main functions of the sample code include: how to create a SymbolTree through the network, and how to insert, delete, and replace the nodes in the SymbolTree. It also includes the modification of the subnet and node replacement through pattern matching.

.. literalinclude:: rewrite_example.py
    :language: python
    :start-at: import

.. automodule:: mindspore.rewrite
    :exclude-members: SparseFunc
    :members:

.. autoclass:: mindspore.rewrite.SparseFunc