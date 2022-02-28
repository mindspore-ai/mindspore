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

mindspore.dataset.text.transforms
---------------------------------

.. mscnnoteautosummary::
    :toctree: dataset_text
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.text.transforms.BasicTokenizer
    mindspore.dataset.text.transforms.BertTokenizer
    mindspore.dataset.text.transforms.CaseFold
    mindspore.dataset.text.transforms.JiebaTokenizer
    mindspore.dataset.text.transforms.Lookup
    mindspore.dataset.text.transforms.Ngram
    mindspore.dataset.text.transforms.NormalizeUTF8
    mindspore.dataset.text.transforms.PythonTokenizer
    mindspore.dataset.text.transforms.RegexReplace
    mindspore.dataset.text.transforms.RegexTokenizer
    mindspore.dataset.text.transforms.SentencePieceTokenizer
    mindspore.dataset.text.transforms.SlidingWindow
    mindspore.dataset.text.transforms.ToNumber
    mindspore.dataset.text.transforms.TruncateSequencePair
    mindspore.dataset.text.transforms.UnicodeCharTokenizer
    mindspore.dataset.text.transforms.UnicodeScriptTokenizer
    mindspore.dataset.text.transforms.WhitespaceTokenizer
    mindspore.dataset.text.transforms.WordpieceTokenizer


mindspore.dataset.text.utils
----------------------------

.. mscnnoteautosummary::
    :toctree: dataset_text
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.text.JiebaMode
    mindspore.dataset.text.NormalizeForm
    mindspore.dataset.text.SentencePieceModel
    mindspore.dataset.text.SentencePieceVocab
    mindspore.dataset.text.SPieceTokenizerLoadType
    mindspore.dataset.text.SPieceTokenizerOutType
    mindspore.dataset.text.to_str
    mindspore.dataset.text.to_bytes
    mindspore.dataset.text.Vocab
