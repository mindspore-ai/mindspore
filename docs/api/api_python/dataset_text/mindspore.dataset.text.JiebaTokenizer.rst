mindspore.dataset.text.JiebaTokenizer
=====================================

.. py:class:: mindspore.dataset.text.JiebaTokenizer(hmm_path, mp_path, mode=JiebaMode.MIX, with_offsets=False)

    使用Jieba分词器对中文字符串进行分词。

    .. note:: 必须保证隐式马尔科夫模型分词（HMMSEgment）和最大概率法分词（MPSegment）所使用的词典文件的完整性。

    参数：
        - **hmm_path** (str) - 隐式马尔科夫模型分词算法使用的词典文件路径，词典可在cppjieba官网获取，
          详见 `cppjieba_github <https://github.com/yanyiwu/cppjieba/tree/master/dict>`_ 。
        - **mp_path** (str) - 最大概率法分词算法使用的词典文件路径，词典可在cppjieba官网获取，
          详见 `cppjieba_github <https://github.com/yanyiwu/cppjieba/tree/master/dict>`_ 。
        - **mode** (:class:`mindspore.dataset.text.JiebaMode` , 可选) - Jieba分词使用的模式，可以取值为JiebaMode.MP、JiebaMode.HMM或JiebaMode.MIX。默认值：JiebaMode.MIX。

          - **JiebaMode.MP**：使用最大概率法算法进行分词。
          - **JiebaMode.HMM**：使用隐马尔可夫模型算法进行分词。
          - **JiebaMode.MIX**：使用隐式马尔科夫模型分词算法和最大概率法分词算法混合进行分词。

        - **with_offsets** (bool, 可选) - 是否输出标记(token)的偏移量。默认值：False。

    异常：      
        - **ValueError** - 没有提供参数 `hmm_path` 或为None。
        - **ValueError** - 没有提供参数 `mp_path` 或为None。
        - **TypeError** - 参数 `hmm_path` 和 `mp_path` 类型不为str。
        - **TypeError** - 参数 `with_offsets` 类型不为bool。

    .. py:method:: add_dict(user_dict)

        将用户定义的词添加到 `JiebaTokenizer` 的字典中。

        参数：
            - **user_dict** (Union[str, dict]) - 有两种输入方式。可以通过指定jieba字典格式的文件路径加载。
              要求的jieba字典格式为：[word，freq]，如：

              .. code-block::

                  word1 freq1
                  word2 None
                  word3 freq3

              在提供的jieba字典文件中，只有有效的词对才会被添加到字典中，无效的输入行将被忽略，且不返回错误或警告状态。
              同时用户也可以通过Python dict定义要添加的词汇，要求的Python字典格式为：{word1:freq1, word2:freq2,...}。

    .. py:method:: add_word(word, freq=None)

        将用户定义的词添加到 JiebaTokenizer 的字典中。

        参数：
            - **word** (str) - 要添加到 `JiebaTokenizer` 词典中的单词，注意通过此接口添加的单词不会被写入本地的模型文件中。
            - **freq** (int，可选) - 要添加的单词的频率。频率越高，单词被分词的机会越大。默认值：None，使用默认频率。
