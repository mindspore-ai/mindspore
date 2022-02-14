mindspore.dataset.text.transforms.SentencePieceTokenizer
========================================================

.. py:class:: mindspore.dataset.text.transforms.SentencePieceTokenizer(mode, out_type)

    使用SentencePiece分词器对字符串进行分词。

    **参数：**

    - **mode** (Union[str, SentencePieceVocab]) - 如果输入是字符串，则代表SentencePiece模型文件的路径；
      如果输入是SentencePieceVocab类型，则代表一个SentencePieceVocab 对象。
    - **out_type** (SPieceTokenizerOutType) - 分词器输出的类型，可以取值为 SPieceTokenizerOutType.STRING 或 SPieceTokenizerOutType.INT。
      
      - SPieceTokenizerOutType.STRING，表示 SentencePice分词器 的输出类型是字符串。
      - SPieceTokenizerOutType.INT，表示 SentencePice分词器 的输出类型是整型。

    **异常：**

    - **TypeError** - 参数 `mode` 的类型不是string或SentencePieceVocab。
    - **TypeError** - 参数 `out_type` 的类型不是SPieceTokenizerOutType。
