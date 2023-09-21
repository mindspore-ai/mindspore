mindspore.dataset.text.SentencePieceVocab
==========================================

.. py:class:: mindspore.dataset.text.SentencePieceVocab

    用于执行分词的SentencePiece对象。

    .. py:method:: from_dataset(dataset, col_names, vocab_size, character_coverage, model_type, params)

        从数据集构建SentencePiece。

        参数：
            - **dataset** (Dataset) - 表示用于构建SentencePiece对象的数据集。
            - **col_names** (list) - 表示列名称的列表。
            - **vocab_size** (int) - 表示词汇大小。
            - **character_coverage** (float) - 表示模型涵盖的字符数量。推荐值： ``0.9995`` ，适用于具有丰富字符集的语言，如日文或中文， ``1.0`` 适用于具有小字符集的其他语言。
            - **model_type** (:class:`~.text.SentencePieceModel`) - 想要使用的子词算法。可选值详见 :class:`~.text.SentencePieceModel` 。
            - **params** (dict) - 表示没有传入参数的字典。

        返回：
            SentencePieceVocab，从数据集构建的Vocab对象。

    .. py:method:: from_file(file_path, vocab_size, character_coverage, model_type, params)

        从文件中构建一个SentencePiece对象。

        参数：
            - **file_path** (list) - 表示包含SentencePiece文件路径的一个列表。
            - **vocab_size** (int) - 表示词汇大小。
            - **character_coverage** (float) - 表示模型涵盖的字符数量。推荐值： ``0.9995`` ，适用于具有丰富字符集的语言，如日文或中文， ``1.0`` 适用于具有小字符集的其他语言。
            - **model_type** (:class:`~.text.SentencePieceModel`) - 想要使用的子词算法。可选值详见 :class:`~.text.SentencePieceModel` 。
            - **params** (dict) - 表示没有传入参数的字典（参数派生自SentencePiece库）。

        返回：
            SentencePieceVocab，表示从文件中构建的Vocab对象。

    .. py:method:: save_model(vocab, path, filename)

        将模型保存到给定的文件路径。

        参数：
            - **vocab** (SentencePieceVocab) - 表示一个SentencePiece对象。
            - **path** (str) - 表示存储模型的路径。
            - **filename** (str) - 表示文件名称。
