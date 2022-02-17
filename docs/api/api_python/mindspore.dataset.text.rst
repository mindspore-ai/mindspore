mindspore.dataset.text
======================

This module is to support text processing for NLP. It includes two parts: transforms and utils. transforms is a high performance NLP text processing module which is developed with ICU4C and cppjieba. utils provides some general methods for NLP text processing.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    from mindspore.dataset import text

mindspore.dataset.text.transforms
---------------------------------

.. msnoteautosummary::
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

.. msnoteautosummary::
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
