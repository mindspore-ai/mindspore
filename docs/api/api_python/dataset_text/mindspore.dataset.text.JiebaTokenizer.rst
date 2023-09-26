mindspore.dataset.text.JiebaTokenizer
=====================================

.. py:class:: mindspore.dataset.text.JiebaTokenizer(hmm_path, mp_path, mode=JiebaMode.MIX, with_offsets=False)

    使用Jieba分词器对中文字符串进行分词。

    .. note::
        隐式马尔可夫模型（Hidden Markov Model）分词和最大概率法（Max Probability）分词所使用的词典文件可通过
        `cppjieba开源仓 <https://github.com/yanyiwu/cppjieba/tree/master/dict>`_ 获取，请保证文件的有效性与
        完整性。

    参数：
        - **hmm_path** (str) - 隐式马尔可夫模型分词所使用的词典文件路径。
        - **mp_path** (str) - 最大概率法分词所使用的词典文件路径。
        - **mode** (:class:`~.text.JiebaMode`, 可选) - 想要使用的分词算法。可选值详见 :class:`~.text.JiebaMode` 。
        - **with_offsets** (bool, 可选) - 是否输出各Token在原字符串中的起始和结束偏移量。默认值： ``False`` 。

    异常：      
        - **TypeError** - 当 `hmm_path` 不为str类型。
        - **TypeError** - 当 `mp_path` 不为str类型。
        - **TypeError** - 当 `mode` 不为 :class:`~.text.JiebaMode` 类型。
        - **TypeError** - 当 `with_offsets` 不为bool类型。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_

    .. py:method:: add_dict(user_dict)

        添加指定的词映射字典到Vocab中。

        参数：
            - **user_dict** (Union[str, dict[str, int]]) - 待添加到Vocab中的词映射。
              若输入类型为str，表示存储待添加词映射的文件路径，文件的每一行需包含两个字段，间隔一个空格，其中第一个
              字段表示词本身，第二个字段需为数字，表示词频。无效的行将被忽略，且不返回错误或告警。
              若输入类型为dict[str, int]，表示存储待添加词映射的字典，其中键名为词本身，键值为词频。

    .. py:method:: add_word(word, freq=None)

        添加一个指定的词映射到Vocab中。

        参数：
            - **word** (str) - 待添加到Vocab中的词。
            - **freq** (int，可选) - 待添加词的词频。词频越高，单词被分词的机会就越大。默认值： ``None`` ，使用默认词频。
