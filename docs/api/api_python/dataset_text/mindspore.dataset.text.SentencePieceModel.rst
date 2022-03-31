mindspore.dataset.text.SentencePieceModel
==========================================

.. py:class:: mindspore.dataset.text.SentencePieceModel

    SentencePiece分词方法的枚举类。

    可选的枚举值包括： `SentencePieceModel.UNIGRAM` 、 `SentencePieceModel.BPE` 、 `SentencePieceModel.CHAR` 和 `SentencePieceModel.WORD` 。

    - **SentencePieceModel.UNIGRAM** - Unigram语言模型意味着句子中的下一个单词被假定为独立于模型生成的前一个单词。
    - **SentencePieceModel.BPE** - 指字节对编码算法，它取代了最频繁的句子对中的字节数，其中包含一个未使用的字节。
    - **SentencePieceModel.CHAR** - 引用基于字符的SentencePiece模型类型。
    - **SentencePieceModel.WORD** - 引用基于单词的SentencePiece模型类型。
    