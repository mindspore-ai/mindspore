mindspore.dataset.text
======================

此模块用于文本数据增强，包括 `transforms` 和 `utils` 两个子模块。

`transforms` 是一个高性能文本数据增强模块，支持常见的文本数据增强处理。

`utils` 提供了一些文本处理的工具方法。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.dataset as ds
    from mindspore.dataset import text

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- TextTensorOperation，所有文本数据处理操作的基类，派生自TensorOperation。

数据增强算子可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式一般用于处理数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore.dataset.html#数据处理pipeline介绍>`_。
- Eager模式一般用于零散样本，文本预处理举例如下：

  .. code-block::

      from mindspore.dataset import text
      from mindspore.dataset.text import NormalizeForm

      # 构造词汇表
      vocab_list = {"床": 1, "前": 2, "明": 3, "月": 4, "光": 5, "疑": 6,
                    "是": 7, "地": 8, "上": 9, "霜": 10, "举": 11, "头": 12,
                    "望": 13, "低": 14, "思": 15, "故": 16, "乡": 17, "繁": 18,
                    "體": 19, "字": 20, "嘿": 21, "哈": 22, "大": 23, "笑": 24,
                    "嘻": 25, "UNK": 26}
      vocab = text.Vocab.from_dict(vocab_list)
      tokenizer_op = text.BertTokenizer(vocab=vocab, suffix_indicator='##', max_bytes_per_token=100,
                                        unknown_token='[UNK]', lower_case=False, keep_whitespace=False,
                                        normalization_form=NormalizeForm.NONE, preserve_unused_token=True,
                                        with_offsets=False)
      # 分词
      tokens = tokenizer_op("床前明月光，疑是地上霜，举头望明月，低头思故乡。")
      print("token: {}".format(tokens), flush=True)

      # 根据单词查找id
      ids = vocab.tokens_to_ids(tokens)
      print("token to id: {}".format(ids), flush=True)

      # 根据id查找单词
      tokens_from_ids = vocab.ids_to_tokens([15, 3, 7])
      print("token to id: {}".format(tokens_from_ids), flush=True)

变换
-----

.. mscnnoteautosummary::
    :toctree: dataset_text
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.text.BasicTokenizer
    mindspore.dataset.text.BertTokenizer
    mindspore.dataset.text.CaseFold
    mindspore.dataset.text.FilterWikipediaXML
    mindspore.dataset.text.JiebaTokenizer
    mindspore.dataset.text.Lookup
    mindspore.dataset.text.Ngram
    mindspore.dataset.text.NormalizeUTF8
    mindspore.dataset.text.PythonTokenizer
    mindspore.dataset.text.RegexReplace
    mindspore.dataset.text.RegexTokenizer
    mindspore.dataset.text.SentencePieceTokenizer
    mindspore.dataset.text.SlidingWindow
    mindspore.dataset.text.ToNumber
    mindspore.dataset.text.ToVectors
    mindspore.dataset.text.TruncateSequencePair
    mindspore.dataset.text.UnicodeCharTokenizer
    mindspore.dataset.text.UnicodeScriptTokenizer
    mindspore.dataset.text.WhitespaceTokenizer
    mindspore.dataset.text.WordpieceTokenizer


工具
-----

.. mscnnoteautosummary::
    :toctree: dataset_text
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.text.CharNGram
    mindspore.dataset.text.FastText
    mindspore.dataset.text.GloVe
    mindspore.dataset.text.JiebaMode
    mindspore.dataset.text.NormalizeForm
    mindspore.dataset.text.SentencePieceModel
    mindspore.dataset.text.SentencePieceVocab
    mindspore.dataset.text.SPieceTokenizerLoadType
    mindspore.dataset.text.SPieceTokenizerOutType
    mindspore.dataset.text.Vectors
    mindspore.dataset.text.Vocab
    mindspore.dataset.text.to_bytes
    mindspore.dataset.text.to_str
