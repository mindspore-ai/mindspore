mindspore.dataset.text.transforms.JiebaTokenizer
================================================

.. py:class:: mindspore.dataset.text.transforms.JiebaTokenizer(hmm_path, mp_path, mode=JiebaMode.MIX, with_offsets=False)

    使用Jieba分词器对中文字符串进行分词。

    .. note:: 必须保证 HMMSEgment 算法和 MPSegment 算法所使用的字典文件的完整性。

    **参数：**

    - **hmm_path** (str) -  HMMSegment 算法使用的字典文件路径，字典可在cppjieba官网获取，
      详见 `cppjieba_github <https://github.com/yanyiwu/cppjieba/tree/master/dict>`_ 。
    - **mp_path** (str) - MPSegment 算法使用的字典文件路径，字典可在cppjieba官网获取，
      详见 `cppjieba_github <https://github.com/yanyiwu/cppjieba/tree/master/dict>`_ 。
    - **mode** (JiebaMode, 可选) - Jieba分词使用的模式，可以取值为 JiebaMode.MP、JiebaMode.HMM 或 JiebaMode.MIX。默认值：JiebaMode.MIX。

      - **JiebaMode.MP**：使用最大概率法算法进行分词。
      - **JiebaMode.HMM**：使用隐马尔可夫模型算法进行分词。
      - **JiebaMode.MIX**：使用 MPSegment 和 HMMSegment 算法混合进行分词。

    - **with_offsets** (bool, 可选) - 是否输出标记(token)的偏移量，默认值：False。

    **异常：**
      
    - **ValueError** - 没有提供参数 `hmm_path` 或为None。
    - **ValueError** - 没有提供参数 `mp_path` 或为None。
    - **TypeError** - 参数 `hmm_path` 和 `mp_path` 类型不为string。
    - **TypeError** - 参数 `with_offsets` 类型不为bool。

.. py:method:: add_word(self, word, freq=None)

    将用户定义的词添加到 JiebaTokenizer 的字典中。

    **参数：**

    - **word** (str) - 要添加到 JiebaTokenizer 词典中的单词，注意通过此接口添加的单词不会被写入本地的模型文件中。
    - **freq** (int，可选) - 要添加的单词的频率。频率越高，单词被分词的机会越大。默认值：None，使用默认频率。

.. py:method:: add_dict(self, user_dict)

    将用户定义的词添加到 JiebaTokenizer 的字典中。

    **参数：**

    - **user_dict** (Union[str, dict]) - 有两种输入方式。可以通过指定jieba字典格式的文件路径加载。
      要求的jieba字典格式为：[word，freq]，如：

      .. code-block::

          word1 freq1
          word2 None
          word3 freq3

      也可以通过Python dict加载，要求的 Python 字典格式为：{word1:freq1, word2:freq2,...}。
      只有用户提供的文件中有效的词对才会被添加到字典中，无的效输入行将被忽略，且不返回错误或警告状态。
