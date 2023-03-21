mindspore.dataset.text.SentencePieceTokenizer
=============================================

.. py:class:: mindspore.dataset.text.SentencePieceTokenizer(mode, out_type)

    使用SentencePiece分词器对字符串进行分词。

    参数：
        - **mode** (Union[str, SentencePieceVocab]) - SentencePiece模型。
          如果输入是字符串类型，则代表要加载的SentencePiece模型文件的路径；
          如果输入是SentencePieceVocab类型，则要求是构造好的 :class:`mindspore.dataset.text.SentencePieceVocab` 对象。
        - **out_type** (:class:`mindspore.dataset.text.SPieceTokenizerOutType`) - 分词器输出的类型，可以取值为 SPieceTokenizerOutType.STRING 或 SPieceTokenizerOutType.INT。
        
          - SPieceTokenizerOutType.STRING，表示SentencePice分词器的输出类型是str。
          - SPieceTokenizerOutType.INT，表示SentencePice分词器的输出类型是int。

    异常：
        - **TypeError** - 参数 `mode` 的类型不是字符串或 :class:`mindspore.dataset.text.SentencePieceVocab` 。
        - **TypeError** - 参数 `out_type` 的类型不是 :class:`mindspore.dataset.text.SPieceTokenizerOutType` 。
