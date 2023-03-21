# Copyright 2019-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This file contains specific text dataset loading classes. You can easily use
these classes to load the prepared dataset. For example:
    IMDBDataset: which is IMDB dataset.
    WikiTextDataset: which is Wiki text dataset.
    CLUEDataset: which is CLUE dataset.
    YelpReviewDataset: which is yelp review dataset.
    ...
After declaring the dataset object, you can further apply dataset operations
(e.g. filter, skip, concat, map, batch) on it.
"""
import mindspore._c_dataengine as cde

from .datasets import TextBaseDataset, SourceDataset, MappableDataset, Shuffle
from .validators import check_imdb_dataset, check_iwslt2016_dataset, check_iwslt2017_dataset, \
    check_penn_treebank_dataset, check_ag_news_dataset, check_amazon_review_dataset, check_udpos_dataset, \
    check_wiki_text_dataset, check_conll2000_dataset, check_cluedataset, \
    check_sogou_news_dataset, check_textfiledataset, check_dbpedia_dataset, check_yelp_review_dataset, \
    check_en_wik9_dataset, check_yahoo_answers_dataset, check_multi30k_dataset, check_squad_dataset, \
    check_sst2_dataset

from ..core.validator_helpers import replace_none


class AGNewsDataset(SourceDataset, TextBaseDataset):
    """
    AG News dataset.

    The generated dataset has three columns: :py:obj:`[index, title, description]` ,
    and the data type of three columns is string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'test' and 'all'. Default: None, all samples.
        num_samples (int, optional): Number of samples (rows) to read. Default: None, reads the full dataset.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If `shuffle` is False, no shuffling will be performed.
            If `shuffle` is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . This
            argument can only be specified when `num_shards` is also specified. Default: None.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> ag_news_dataset_dir = "/path/to/ag_news_dataset_file"
        >>> dataset = ds.AGNewsDataset(dataset_dir=ag_news_dataset_dir, usage='all')

    About AGNews dataset:

    AG is a collection of over 1 million news articles. The news articles were collected
    by ComeToMyHead from over 2,000 news sources in over 1 year of activity. ComeToMyHead
    is an academic news search engine that has been in operation since July 2004.
    The dataset is provided by academics for research purposes such as data mining
    (clustering, classification, etc.), information retrieval (ranking, searching, etc.),
    xml, data compression, data streaming, and any other non-commercial activities.
    AG's news topic classification dataset was constructed by selecting the four largest
    classes from the original corpus. Each class contains 30,000 training samples and
    1,900 test samples. The total number of training samples in train.csv is 120,000
    and the number of test samples in test.csv is 7,600.

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── ag_news_dataset_dir
            ├── classes.txt
            ├── train.csv
            ├── test.csv
            └── readme.txt

    Citation:

    .. code-block::

        @misc{zhang2015characterlevel,
        title={Character-level Convolutional Networks for Text Classification},
        author={Xiang Zhang and Junbo Zhao and Yann LeCun},
        year={2015},
        eprint={1509.01626},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
        }
    """

    @check_ag_news_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None,
                 num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.AGNewsNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                              self.shard_id)


class AmazonReviewDataset(SourceDataset, TextBaseDataset):
    """
    Amazon Review Polarity and Amazon Review Full datasets.

    The generated dataset has three columns: :py:obj:`[label, title, content]` ,
    and the data type of three columns is string.

    Args:
        dataset_dir (str): Path to the root directory that contains the Amazon Review Polarity dataset
            or the Amazon Review Full dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'.
            For Polarity dataset, 'train' will read from 3,600,000 train samples,
            'test' will read from 400,000 test samples,
            'all' will read from all 4,000,000 samples.
            For Full dataset, 'train' will read from 3,000,000 train samples,
            'test' will read from 650,000 test samples,
            'all' will read from all 3,650,000 samples. Default: None, all samples.
        num_samples (int, optional): Number of samples (rows) to be read. Default: None, reads the full dataset.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If `shuffle` is False, no shuffling will be performed.
            If `shuffle` is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> amazon_review_dataset_dir = "/path/to/amazon_review_dataset_dir"
        >>> dataset = ds.AmazonReviewDataset(dataset_dir=amazon_review_dataset_dir, usage='all')

    About AmazonReview Dataset:

    The Amazon reviews full dataset consists of reviews from Amazon. The data span a period of 18 years, including ~35
    million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review.
    The dataset is mainly used for text classification, given the content and title, predict the correct star rating.

    The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, 4 and 5 as positive.
    Samples of score 3 is ignored.

    The Amazon Reviews Polarity and Amazon Reviews Full datasets have the same directory structures.
    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── amazon_review_dir
             ├── train.csv
             ├── test.csv
             └── readme.txt

    Citation:

    .. code-block::

        @article{zhang2015character,
          title={Character-level convolutional networks for text classification},
          author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          journal={Advances in neural information processing systems},
          volume={28},
          pages={649--657},
          year={2015}
        }
    """

    @check_amazon_review_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.AmazonReviewNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                    self.shard_id)


class CLUEDataset(SourceDataset, TextBaseDataset):
    """
    CLUE(Chinese Language Understanding Evaluation) dataset.
    Supported CLUE classification tasks: 'AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC' and 'CSL'.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for
            a pattern of files. The list will be sorted in a lexicographical order.
        task (str, optional): The kind of task, one of 'AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC' and 'CSL'.
            Default: 'AFQMC'.
        usage (str, optional): Specify the 'train', 'test' or 'eval' part of dataset. Default: 'train'.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will include all images.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: Shuffle.GLOBAL. Bool type and Shuffle enum are both supported to pass in.
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by mindspore.dataset.Shuffle.

            - Shuffle.GLOBAL: Shuffle both the files and samples, same as setting shuffle to True.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    The generated dataset with different task setting has different output columns:

    +-------------------------+------------------------------+-----------------------------+
    | `task`                  |   `usage`                    |   Output column             |
    +=========================+==============================+=============================+
    | AFQMC                   |   train                      |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    +-------------------------+------------------------------+-----------------------------+
    | TNEWS                   |   train                      |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         |                              |                             |
    |                         |                              |   [keywords, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [label, dtype=uint32]     |
    |                         |                              |                             |
    |                         |                              |   [keywords, dtype=string]  |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         |                              |                             |
    |                         |                              |   [keywords, dtype=string]  |
    +-------------------------+------------------------------+-----------------------------+
    | IFLYTEK                 |   train                      |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [label, dtype=string]     |
    |                         |                              |                             |
    |                         |                              |   [label_des, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence, dtype=string]  |
    +-------------------------+------------------------------+-----------------------------+
    | CMNLI                   |   train                      |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [sentence1, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [sentence2, dtype=string] |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    +-------------------------+------------------------------+-----------------------------+
    | WSC                     |   train                      |  [span1_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span2_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span1_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [span2_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [idx, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |  [text, dtype=string]       |
    |                         |                              |                             |
    |                         |                              |  [label, dtype=string]      |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |  [span1_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span2_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span1_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [span2_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [idx, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |  [text, dtype=string]       |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |  [span1_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span2_index, dtype=uint32]|
    |                         |                              |                             |
    |                         |                              |  [span1_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [span2_text, dtype=string] |
    |                         |                              |                             |
    |                         |                              |  [idx, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |  [text, dtype=string]       |
    |                         |                              |                             |
    |                         |                              |  [label, dtype=string]      |
    +-------------------------+------------------------------+-----------------------------+
    | CSL                     |   train                      |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [abst, dtype=string]      |
    |                         |                              |                             |
    |                         |                              |   [keyword, dtype=string]   |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    |                         +------------------------------+-----------------------------+
    |                         |   test                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [abst, dtype=string]      |
    |                         |                              |                             |
    |                         |                              |   [keyword, dtype=string]   |
    |                         +------------------------------+-----------------------------+
    |                         |   eval                       |   [id, dtype=uint32]        |
    |                         |                              |                             |
    |                         |                              |   [abst, dtype=string]      |
    |                         |                              |                             |
    |                         |                              |   [keyword, dtype=string]   |
    |                         |                              |                             |
    |                         |                              |   [label, dtype=string]     |
    +-------------------------+------------------------------+-----------------------------+

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: task is not in 'AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC' or 'CSL'.
        ValueError: usage is not in 'train', 'test' or 'eval'.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.

    Examples:
        >>> clue_dataset_dir = ["/path/to/clue_dataset_file"] # contains 1 or multiple clue files
        >>> dataset = ds.CLUEDataset(dataset_files=clue_dataset_dir, task='AFQMC', usage='train')

    About CLUE dataset:

    CLUE, a Chinese Language Understanding Evaluation benchmark. It contains multiple
    tasks, including single-sentence classification, sentence pair classification, and machine
    reading comprehension.

    You can unzip the dataset files into the following structure and read by MindSpore's API,
    such as afqmc dataset:

    .. code-block::

        .
        └── afqmc_public
             ├── train.json
             ├── test.json
             └── dev.json

    Citation:

    .. code-block::

        @article{CLUEbenchmark,
        title   = {CLUE: A Chinese Language Understanding Evaluation Benchmark},
        author  = {Liang Xu, Xuanwei Zhang, Lu Li, Hai Hu, Chenjie Cao, Weitang Liu, Junyi Li, Yudong Li,
                Kai Sun, Yechen Xu, Yiming Cui, Cong Yu, Qianqian Dong, Yin Tian, Dian Yu, Bo Shi, Jun Zeng,
                Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou,
                Shaoweihua Liu, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Zhenzhong Lan},
        journal = {arXiv preprint arXiv:2004.05986},
        year    = {2020},
        howpublished = {https://github.com/CLUEbenchmark/CLUE}
        }
    """

    @check_cluedataset
    def __init__(self, dataset_files, task='AFQMC', usage='train', num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.usage = replace_none(usage, 'train')
        self.task = replace_none(task, 'AFQMC')

    def parse(self, children=None):
        return cde.CLUENode(self.dataset_files, self.task, self.usage, self.num_samples, self.shuffle_flag,
                            self.num_shards, self.shard_id)


class CoNLL2000Dataset(SourceDataset, TextBaseDataset):
    """
    CoNLL-2000(Conference on Computational Natural Language Learning) chunking dataset.

    The generated dataset has three columns: :py:obj:`[word, pos_tag, chunk_tag]` .
    The tensors of column :py:obj:`word` , column :py:obj:`pos_tag` ,
    and column :py:obj:`chunk_tag` are of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the CoNLL2000 chunking dataset.
        usage (str, optional): Usage of dataset, can be 'train', 'test', or 'all'.
            For dataset, 'train' will read from 8,936 train samples,
            'test' will read from 2,012 test samples,
            'all' will read from all 1,0948 samples. Default: None, read all samples.
        num_samples (int, optional): Number of samples (rows) to be read. Default: None, read the full dataset.
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: `mindspore.dataset.Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by mindspore.dataset.Shuffle.

            - Shuffle.GLOBAL: Shuffle both the files and samples, same as setting shuffle to True.
            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into.
            When this argument is specified, `num_samples` reflects the max sample number of per shard. Default: None.
        shard_id (int, optional): The shard ID within `num_shards` . This
            argument can only be specified when `num_shards` is also specified. Default: None.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.

    Examples:
        >>> conll2000_dataset_dir = "/path/to/conll2000_dataset_dir"
        >>> dataset = ds.CoNLL2000Dataset(dataset_dir=conll2000_dataset_dir, usage='all')

    About CoNLL2000 Dataset:

    The CoNLL2000 chunking dataset consists of the text from sections 15-20 of the Wall Street Journal corpus.
    Texts are chunked using IOB notation, and the chunk type has NP, VP, PP, ADJP and ADVP.
    The dataset consist of three columns separated by spaces. The first column contains the current word,
    the second is part-of-speech tag as derived by the Brill tagger and the third is chunk tag as derived from
    the WSJ corpus. Text chunking consists of dividing a text in syntactically correlated parts of words.

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── conll2000_dataset_dir
             ├── train.txt
             ├── test.txt
             └── readme.txt

    Citation:

    .. code-block::

        @inproceedings{tksbuchholz2000conll,
        author     = {Tjong Kim Sang, Erik F. and Sabine Buchholz},
        title      = {Introduction to the CoNLL-2000 Shared Task: Chunking},
        editor     = {Claire Cardie and Walter Daelemans and Claire Nedellec and Tjong Kim Sang, Erik},
        booktitle  = {Proceedings of CoNLL-2000 and LLL-2000},
        publisher  = {Lisbon, Portugal},
        pages      = {127--132},
        year       = {2000}
        }
    """

    @check_conll2000_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.CoNLL2000Node(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                 self.shard_id)


class DBpediaDataset(SourceDataset, TextBaseDataset):
    """
    DBpedia dataset.

    The generated dataset has three columns :py:obj:`[class, title, content]` ,
    and the data type of three columns is string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'.
            'train' will read from 560,000 train samples,
            'test' will read from 70,000 test samples,
            'all' will read from all 630,000 samples. Default: None, all samples.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will include all text.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> dbpedia_dataset_dir = "/path/to/dbpedia_dataset_directory"
        >>>
        >>> # 1) Read 3 samples from DBpedia dataset
        >>> dataset = ds.DBpediaDataset(dataset_dir=dbpedia_dataset_dir, num_samples=3)
        >>>
        >>> # 2) Read train samples from DBpedia dataset
        >>> dataset = ds.DBpediaDataset(dataset_dir=dbpedia_dataset_dir, usage="train")

    About DBpedia dataset:

    The DBpedia dataset consists of 630,000 text samples in 14 classes, there are 560,000 samples in the train.csv
    and 70,000 samples in the test.csv.
    The 14 different classes represent Company, EducationaInstitution, Artist, Athlete, OfficeHolder,
    MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.

    Here is the original DBpedia dataset structure.
    You can unzip the dataset files into this directory structure and read by Mindspore's API.

    .. code-block::

        .
        └── dbpedia_dataset_dir
            ├── train.csv
            ├── test.csv
            ├── classes.txt
            └── readme.txt

    Citation:

    .. code-block::

        @article{DBpedia,
        title   = {DBPedia Ontology Classification Dataset},
        author  = {Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas,
                Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef,
                    Sören Auer, Christian Bizer},
        year    = {2015},
        howpublished = {http://dbpedia.org}
        }
    """

    @check_dbpedia_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.DBpediaNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                               self.shard_id)


class EnWik9Dataset(SourceDataset, TextBaseDataset):
    """
    EnWik9 dataset.

    The generated dataset has one column :py:obj:`[text]` with type string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will include all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: True.
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> en_wik9_dataset_dir = "/path/to/en_wik9_dataset"
        >>> dataset2 = ds.EnWik9Dataset(dataset_dir=en_wik9_dataset_dir, num_samples=2,
        ...                             shuffle=True)

    About EnWik9 dataset:

    The data of EnWik9 is UTF-8 encoded XML consisting primarily of English text. It contains 243,426 article titles,
    of which 85,560 are #REDIRECT to fix broken links, and the rest are regular articles.

    The data is UTF-8 clean. All characters are in the range U'0000 to U'10FFFF with valid encodings of 1 to
    4 bytes. The byte values 0xC0, 0xC1, and 0xF5-0xFF never occur. Also, in the Wikipedia dumps,
    there are no control characters in the range 0x00-0x1F except for 0x09 (tab) and 0x0A (linefeed).
    Linebreaks occur only on paragraph boundaries, so they always have a semantic purpose.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── EnWik9
             ├── enwik9

    Citation:

    .. code-block::

        @NetworkResource{Hutter_prize,
        author    = {English Wikipedia},
        url       = "https://cs.fit.edu/~mmahoney/compression/textdata.html",
        month     = {March},
        year      = {2006}
        }
    """

    @check_en_wik9_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=True,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.EnWik9Node(self.dataset_dir, self.num_samples, self.shuffle_flag, self.num_shards,
                              self.shard_id)


class IMDBDataset(MappableDataset, TextBaseDataset):
    """
    IMDb(Internet Movie Database) dataset.

    The generated dataset has two columns: :py:obj:`[text, label]` .
    The tensor of column :py:obj:`text` is of the string type.
    The column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'.
            Default: None, will read all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will include all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - The shape of the test column.
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> imdb_dataset_dir = "/path/to/imdb_dataset_directory"
        >>>
        >>> # 1) Read all samples (text files) in imdb_dataset_dir with 8 threads
        >>> dataset = ds.IMDBDataset(dataset_dir=imdb_dataset_dir, num_parallel_workers=8)
        >>>
        >>> # 2) Read train samples (text files).
        >>> dataset = ds.IMDBDataset(dataset_dir=imdb_dataset_dir, usage="train")

    About IMDBDataset:

    The IMDB dataset contains 50, 000 highly polarized reviews from the Internet Movie Database (IMDB). The dataset
    was divided into 25 000 comments for training and 25 000 comments for testing, with both the training set and test
    set containing 50% positive and 50% negative comments. Train labels and test labels are all lists of 0 and 1, where
    0 stands for negative and 1 for positive.

    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── imdb_dataset_directory
             ├── train
             │    ├── pos
             │    │    ├── 0_9.txt
             │    │    ├── 1_7.txt
             │    │    ├── ...
             │    ├── neg
             │    │    ├── 0_3.txt
             │    │    ├── 1_1.txt
             │    │    ├── ...
             ├── test
             │    ├── pos
             │    │    ├── 0_10.txt
             │    │    ├── 1_10.txt
             │    │    ├── ...
             │    ├── neg
             │    │    ├── 0_2.txt
             │    │    ├── 1_3.txt
             │    │    ├── ...

    Citation:

    .. code-block::

        @InProceedings{maas-EtAl:2011:ACL-HLT2011,
          author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan
                        and  Ng, Andrew Y.  and  Potts, Christopher},
          title     = {Learning Word Vectors for Sentiment Analysis},
          booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics:
                        Human Language Technologies},
          month     = {June},
          year      = {2011},
          address   = {Portland, Oregon, USA},
          publisher = {Association for Computational Linguistics},
          pages     = {142--150},
          url       = {http://www.aclweb.org/anthology/P11-1015}
        }
    """

    @check_imdb_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.IMDBNode(self.dataset_dir, self.usage, self.sampler)


class IWSLT2016Dataset(SourceDataset, TextBaseDataset):
    """
    IWSLT2016(International Workshop on Spoken Language Translation) dataset.

    The generated dataset has two columns: :py:obj:`[text, translation]` .
    The tensor of column :py:obj: `text` is of the string type.
    The column :py:obj: `translation` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'valid', 'test' and 'all'. Default: None, all samples.
        language_pair (sequence, optional): Sequence containing source and target language, supported values are
            ('en', 'fr'), ('en', 'de'), ('en', 'cs'), ('en', 'ar'), ('fr', 'en'), ('de', 'en'), ('cs', 'en'),
            ('ar', 'en'). Default: ('de', 'en').
        valid_set (str, optional): A string to identify validation set, when usage is valid or all, the validation set
            of `valid_set` type will be read, supported values are 'dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013'
            and 'tst2014'. Default: 'tst2013'.
        test_set (str, optional): A string to identify test set, when usage is test or all, the test set of `test_set`
            type will be read, supported values are 'dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013' and 'tst2014'.
            Default: 'tst2014'.
        num_samples (int, optional): Number of samples (rows) to read. Default: None, reads the full dataset.
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If `shuffle` is False, no shuffling will be performed.
            If `shuffle` is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> iwslt2016_dataset_dir = "/path/to/iwslt2016_dataset_dir"
        >>> dataset = ds.IWSLT2016Dataset(dataset_dir=iwslt2016_dataset_dir, usage='all',
        ...                               language_pair=('de', 'en'), valid_set='tst2013', test_set='tst2014')

    About IWSLT2016 dataset:

    IWSLT is an international oral translation conference, a major annual scientific conference dedicated to all aspects
    of oral translation. The MT task of the IWSLT evaluation activity constitutes a dataset, which can be publicly
    obtained through the WIT3 website `wit3 <https://wit3.fbk.eu>`_ . The IWSLT2016 dataset includes translations from
    English to Arabic, Czech, French, and German, and translations from Arabic, Czech, French, and German to English.

    You can unzip the original IWSLT2016 dataset files into this directory structure and read by MindSpore's API. After
    decompression, you also need to decompress the dataset to be read in the specified folder. For example, if you want
    to read the dataset of de-en, you need to unzip the tgz file in the de/en directory, the dataset is in the
    unzipped folder.

    .. code-block::

        .
        └── iwslt2016_dataset_directory
             ├── subeval_files
             └── texts
                  ├── ar
                  │    └── en
                  │        └── ar-en
                  ├── cs
                  │    └── en
                  │        └── cs-en
                  ├── de
                  │    └── en
                  │        └── de-en
                  │            ├── IWSLT16.TED.dev2010.de-en.de.xml
                  │            ├── train.tags.de-en.de
                  │            ├── ...
                  ├── en
                  │    ├── ar
                  │    │   └── en-ar
                  │    ├── cs
                  │    │   └── en-cs
                  │    ├── de
                  │    │   └── en-de
                  │    └── fr
                  │        └── en-fr
                  └── fr
                       └── en
                           └── fr-en

    Citation:

    .. code-block::

        @inproceedings{cettoloEtAl:EAMT2012,
        Address = {Trento, Italy},
        Author = {Mauro Cettolo and Christian Girardi and Marcello Federico},
        Booktitle = {Proceedings of the 16$^{th}$ Conference of the European Association for Machine Translation
                     (EAMT)},
        Date = {28-30},
        Month = {May},
        Pages = {261--268},
        Title = {WIT$^3$: Web Inventory of Transcribed and Translated Talks},
        Year = {2012}}
    """

    @check_iwslt2016_dataset
    def __init__(self, dataset_dir, usage=None, language_pair=None, valid_set=None, test_set=None,
                 num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, num_parallel_workers=None,
                 cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')
        self.language_pair = replace_none(language_pair, ["de", "en"])
        self.valid_set = replace_none(valid_set, 'tst2013')
        self.test_set = replace_none(test_set, 'tst2014')

    def parse(self, children=None):
        return cde.IWSLT2016Node(self.dataset_dir, self.usage, self.language_pair, self.valid_set, self.test_set,
                                 self.num_samples, self.shuffle_flag, self.num_shards, self.shard_id)


class IWSLT2017Dataset(SourceDataset, TextBaseDataset):
    """
    IWSLT2017(International Workshop on Spoken Language Translation) dataset.

    The generated dataset has two columns: :py:obj:`[text, translation]` .
    The tensor of column :py:obj:`text` and :py:obj:`translation` are of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'valid', 'test' and 'all'. Default: None, all samples.
        language_pair (sequence, optional): List containing src and tgt language, supported values are ('en', 'nl'),
            ('en', 'de'), ('en', 'it'), ('en', 'ro'), ('nl', 'en'), ('nl', 'de'), ('nl', 'it'), ('nl', 'ro'),
            ('de', 'en'), ('de', 'nl'), ('de', 'it'), ('de', 'ro'), ('it', 'en'), ('it', 'nl'), ('it', 'de'),
            ('it', 'ro'), ('ro', 'en'), ('ro', 'nl'), ('ro', 'de'), ('ro', 'it'). Default: ('de', 'en').
        num_samples (int, optional): Number of samples (rows) to read. Default: None, reads the full dataset.
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> iwslt2017_dataset_dir = "/path/to/iwslt2017_dataset_dir"
        >>> dataset = ds.IWSLT2017Dataset(dataset_dir=iwslt2017_dataset_dir, usage='all', language_pair=('de', 'en'))

    About IWSLT2017 dataset:

    IWSLT is an international oral translation conference, a major annual scientific conference dedicated to all aspects
    of oral translation. The MT task of the IWSLT evaluation activity constitutes a dataset, which can be publicly
    obtained through the WIT3 website  `wit3 <https://wit3.fbk.eu>`_ . The IWSLT2017 dataset involves German, English,
    Italian, Dutch, and Romanian. The dataset includes translations in any two different languages.

    You can unzip the original IWSLT2017 dataset files into this directory structure and read by MindSpore's API. You
    need to decompress the dataset package in texts/DeEnItNlRo/DeEnItNlRo directory to get the DeEnItNlRo-DeEnItNlRo
    subdirectory.

    .. code-block::

        .
        └── iwslt2017_dataset_directory
            └── DeEnItNlRo
                └── DeEnItNlRo
                    └── DeEnItNlRo-DeEnItNlRo
                        ├── IWSLT17.TED.dev2010.de-en.de.xml
                        ├── train.tags.de-en.de
                        ├── ...

    Citation:

    .. code-block::

        @inproceedings{cettoloEtAl:EAMT2012,
        Address = {Trento, Italy},
        Author = {Mauro Cettolo and Christian Girardi and Marcello Federico},
        Booktitle = {Proceedings of the 16$^{th}$ Conference of the European Association for Machine Translation
                     (EAMT)},
        Date = {28-30},
        Month = {May},
        Pages = {261--268},
        Title = {WIT$^3$: Web Inventory of Transcribed and Translated Talks},
        Year = {2012}}
    """

    @check_iwslt2017_dataset
    def __init__(self, dataset_dir, usage=None, language_pair=None, num_samples=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')
        self.language_pair = replace_none(language_pair, ["de", "en"])

    def parse(self, children=None):
        return cde.IWSLT2017Node(self.dataset_dir, self.usage, self.language_pair, self.num_samples,
                                 self.shuffle_flag, self.num_shards, self.shard_id)


class Multi30kDataset(SourceDataset, TextBaseDataset):
    """
    Multi30k dataset.

    The generated dataset has two columns :py:obj:`[text, translation]` .
    The tensor of column :py:obj:`text` is of the string type.
    The tensor of column :py:obj:`translation` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'test, 'valid' or 'all'.
            Default: None, will read all samples.
        language_pair (Sequence[str, str], optional): Acceptable language_pair include ['en', 'de'], ['de', 'en'].
            Default: None, means ['en', 'de'].
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Whether to shuffle the dataset. Default: None, means Shuffle.GLOBAL.
            If False is provided, no shuffling will be performed.
            If True is provided, it is the same as setting to mindspore.dataset.Shuffle.GLOBAL.
            If Shuffle is provided, the effect is as follows:

            - Shuffle.GLOBAL: Shuffle both the files and samples.
            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `usage` is not 'train', 'test', 'valid' or 'all'.
        TypeError: If `language_pair` is not of type Sequence[str, str].
        RuntimeError: If num_samples is less than 0.
        RuntimeError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> multi30k_dataset_dir = "/path/to/multi30k_dataset_directory"
        >>> data = ds.Multi30kDataset(dataset_dir=multi30k_dataset_dir, usage='all', language_pair=['de', 'en'])

    About Multi30k dataset:

    Multi30K is a multilingual dataset that features approximately 31,000 standardized images
    described in multiple languages. The images are sourced from Flickr and each image comes
    with sentence descripitions in both English and German, as well as descriptions in other
    languages. Multi30k is used primarily for training and testing in tasks such as image
    captioning, machine translation, and visual question answering.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        └── multi30k_dataset_directory
              ├── training
              │    ├── train.de
              │    └── train.en
              ├── validation
              │    ├── val.de
              │    └── val.en
              └── mmt16_task1_test
                   ├── val.de
                   └── val.en

    Citation:

    .. code-block::

        @article{elliott-EtAl:2016:VL16,
        author    = {{Elliott}, D. and {Frank}, S. and {Sima'an}, K. and {Specia}, L.},
        title     = {Multi30K: Multilingual English-German Image Descriptions},
        booktitle = {Proceedings of the 5th Workshop on Vision and Language},
        year      = {2016},
        pages     = {70--74},
        year      = 2016
        }
    """

    @check_multi30k_dataset
    def __init__(self, dataset_dir, usage=None, language_pair=None, num_samples=None,
                 num_parallel_workers=None, shuffle=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')
        self.language_pair = replace_none(language_pair, ["en", "de"])
        self.shuffle = replace_none(shuffle, Shuffle.GLOBAL)

    def parse(self, children=None):
        return cde.Multi30kNode(self.dataset_dir, self.usage, self.language_pair, self.num_samples,
                                self.shuffle_flag, self.num_shards, self.shard_id)


class PennTreebankDataset(SourceDataset, TextBaseDataset):
    """
    PennTreebank dataset.

    The generated dataset has one column :py:obj:`[text]` .
    The tensor of column :py:obj:`text` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'test', 'valid' and 'all'.
            'train' will read from 42,068 train samples of string type,
            'test' will read from 3,370 test samples of string type,
            'valid' will read from 3,761 test samples of string type,
            'all' will read from all 49,199 samples of string type. Default: None, all samples.
        num_samples (int, optional): Number of samples (rows) to read. Default: None, reads the full dataset.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> penn_treebank_dataset_dir = "/path/to/penn_treebank_dataset_directory"
        >>> dataset = ds.PennTreebankDataset(dataset_dir=penn_treebank_dataset_dir, usage='all')

    About PennTreebank dataset:

    Penn Treebank (PTB) dataset, is widely used in machine learning for NLP (Natural Language Processing)
    research. Word-level PTB does not contain capital letters, numbers, and punctuations, and the vocabulary
    is capped at 10k unique words, which is relatively small in comparison to most modern datasets which
    can result in a larger number of out of vocabulary tokens.

    Here is the original PennTreebank dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── PennTreebank_dataset_dir
             ├── ptb.test.txt
             ├── ptb.train.txt
             └── ptb.valid.txt

    Citation:

    .. code-block::

        @techreport{Santorini1990,
          added-at = {2014-03-26T23:25:56.000+0100},
          author = {Santorini, Beatrice},
          biburl = {https://www.bibsonomy.org/bibtex/234cdf6ddadd89376090e7dada2fc18ec/butonic},
          file = {:Santorini - Penn Treebank tag definitions.pdf:PDF},
          institution = {Department of Computer and Information Science, University of Pennsylvania},
          interhash = {818e72efd9e4b5fae3e51e88848100a0},
          intrahash = {34cdf6ddadd89376090e7dada2fc18ec},
          keywords = {dis pos tagging treebank},
          number = {MS-CIS-90-47},
          timestamp = {2014-03-26T23:25:56.000+0100},
          title = {Part-of-speech tagging guidelines for the {P}enn {T}reebank {P}roject},
          url = {ftp://ftp.cis.upenn.edu/pub/treebank/doc/tagguide.ps.gz},
          year = 1990
        }
    """

    @check_penn_treebank_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.PennTreebankNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                    self.shard_id)


class SogouNewsDataset(SourceDataset, TextBaseDataset):
    r"""
    Sogou News dataset.

    The generated dataset has three columns: :py:obj:`[index, title, content]` ,
    and the data type of three columns is string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all' .
            'train' will read from 450,000 train samples, 'test' will read from 60,000 test samples,
            'all' will read from all 510,000 samples. Default: None, all samples.
        num_samples (int, optional): Number of samples (rows) to read. Default: None, read all samples.
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples, same as setting shuffle to True.

            - Shuffle.FILES: Shuffle files only.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> sogou_news_dataset_dir = "/path/to/sogou_news_dataset_dir"
        >>> dataset = ds.SogouNewsDataset(dataset_dir=sogou_news_dataset_dir, usage='all')

    About SogouNews Dataset:

    SogouNews dataset includes 3 columns, corresponding to class index (1 to 5), title and content. The title and
    content are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes ("").
    New lines are escaped by a backslash followed with an "n" character, that is "\n".

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── sogou_news_dir
             ├── classes.txt
             ├── readme.txt
             ├── test.csv
             └── train.csv

    Citation:

    .. code-block::

        @misc{zhang2015characterlevel,
            title={Character-level Convolutional Networks for Text Classification},
            author={Xiang Zhang and Junbo Zhao and Yann LeCun},
            year={2015},
            eprint={1509.01626},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """

    @check_sogou_news_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.SogouNewsNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                                 self.num_shards, self.shard_id)


class SQuADDataset(SourceDataset, TextBaseDataset):
    """
    SQuAD 1.1 and SQuAD 2.0 datasets.

    The generated dataset with different versions and usages has the same output columns:
    :py:obj:`[context, question, text, answer_start]` .
    The tensor of column :py:obj:`context` is of the string type.
    The tensor of column :py:obj:`question` is of the string type.
    The tensor of column :py:obj:`text` is the answer in the context of the string type.
    The tensor of column :py:obj:`answer_start` is the start index of answer in context,
    which is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Specify the 'train', 'dev' or 'all' part of dataset. Default: None, all samples.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will include all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Whether to shuffle the dataset. Default: Shuffle.GLOBAL.
            If False is provided, no shuffling will be performed.
            If True is provided, it is the same as setting to mindspore.dataset.Shuffle.GLOBAL.
            If Shuffle is provided, the effect is as follows:

            - Shuffle.GLOBAL: Shuffle both the files and samples.
            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> squad_dataset_dir = "/path/to/squad_dataset_file"
        >>> dataset = ds.SQuADDataset(dataset_dir=squad_dataset_dir, usage='all')

    About SQuAD dataset:

    SQuAD (Stanford Question Answering Dataset) is a reading comprehension dataset, consisting of questions posed by
    crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span,
    from the corresponding reading passage, or the question might be unanswerable.

    SQuAD 1.1, the previous version of the SQuAD dataset, contains 100,000+ question-answer pairs on 500+ articles.
    SQuAD 2.0 combines the 100,000 questions in SQuAD 1.1 with over 50,000 unanswerable questions written adversarially
    by crowdworkers to look similar to answerable ones. To do well on SQuAD 2.0, systems must not only answer questions
    when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

    You can get the dataset files into the following structure and read by MindSpore's API,

    For SQuAD 1.1:

    .. code-block::

        .
        └── SQuAD1
             ├── train-v1.1.json
             └── dev-v1.1.json

    For SQuAD 2.0:

    .. code-block::

        .
        └── SQuAD2
             ├── train-v2.0.json
             └── dev-v2.0.json

    Citation:

    .. code-block::

        @misc{rajpurkar2016squad,
            title         = {SQuAD: 100,000+ Questions for Machine Comprehension of Text},
            author        = {Pranav Rajpurkar and Jian Zhang and Konstantin Lopyrev and Percy Liang},
            year          = {2016},
            eprint        = {1606.05250},
            archivePrefix = {arXiv},
            primaryClass  = {cs.CL}
        }

        @misc{rajpurkar2018know,
            title         = {Know What You Don't Know: Unanswerable Questions for SQuAD},
            author        = {Pranav Rajpurkar and Robin Jia and Percy Liang},
            year          = {2018},
            eprint        = {1806.03822},
            archivePrefix = {arXiv},
            primaryClass  = {cs.CL}
        }
    """

    @check_squad_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.SQuADNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                             self.num_shards, self.shard_id)


class SST2Dataset(SourceDataset, TextBaseDataset):
    """
    SST2(Stanford Sentiment Treebank v2) dataset.

    The generated dataset's train.tsv and dev.tsv have two columns :py:obj:`[sentence, label]` .
    The generated dataset's test.tsv has one column :py:obj:`[sentence]` .
    The tensor of column :py:obj:`sentence` and :py:obj:`label` are of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `dev`. `train` will read
            from 67,349 train samples, `test` will read from 1,821 test samples, `dev` will read from
            all 872 samples. Default: None, will read train samples.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will include all text.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle the samples.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards. This argument can only be specified when
            num_shards is also specified. Default: None.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but shard_id is None.
        RuntimeError: If `shard_id` is specified but num_shards is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> sst2_dataset_dir = "/path/to/sst2_dataset_directory"
        >>>
        >>> # 1) Read 3 samples from SST2 dataset
        >>> dataset = ds.SST2Dataset(dataset_dir=sst2_dataset_dir, num_samples=3)
        >>>
        >>> # 2) Read train samples from SST2 dataset
        >>> dataset = ds.SST2Dataset(dataset_dir=sst2_dataset_dir, usage="train")

    About SST2 dataset:
    The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows for a complete
    analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced
    by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews. It was parsed
    with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each
    annotated by 3 human judges.

    Here is the original SST2 dataset structure.
    You can unzip the dataset files into this directory structure and read by Mindspore's API.

    .. code-block::

        .
        └── sst2_dataset_dir
            ├── train.tsv
            ├── test.tsv
            ├── dev.tsv
            └── original

    Citation:

    .. code-block::

        @inproceedings{socher-etal-2013-recursive,
            title     = {Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank},
            author    = {Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning,
                          Christopher D. and Ng, Andrew and Potts, Christopher},
            booktitle = {Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},
            month     = oct,
            year      = {2013},
            address   = {Seattle, Washington, USA},
            publisher = {Association for Computational Linguistics},
            url       = {https://www.aclweb.org/anthology/D13-1170},
            pages     = {1631--1642},
        }
    """

    @check_sst2_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "train")

    def parse(self, children=None):
        return cde.SST2Node(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                            self.num_shards, self.shard_id)


class TextFileDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses datasets stored on disk in text format.
    The generated dataset has one column :py:obj:`[text]` with type string.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will include all images.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: `Shuffle.GLOBAL` . Bool type and Shuffle enum are both supported to pass in.
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by mindspore.dataset.Shuffle.

            - Shuffle.GLOBAL: Shuffle both the files and samples, same as setting shuffle to True.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> text_file_dataset_dir = ["/path/to/text_file_dataset_file"] # contains 1 or multiple text files
        >>> dataset = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
    """

    @check_textfiledataset
    def __init__(self, dataset_files, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

    def parse(self, children=None):
        return cde.TextFileNode(self.dataset_files, self.num_samples, self.shuffle_flag, self.num_shards,
                                self.shard_id)


class UDPOSDataset(SourceDataset, TextBaseDataset):
    """
    UDPOS(Universal Dependencies dataset for Part of Speech) dataset.

    The generated dataset has three columns: :py:obj:`[word, universal, stanford]` ,
    and the data type of three columns is string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test', 'valid' or 'all'. 'train' will read from
            12,543 train samples, 'test' will read from 2,077 test samples, 'valid' will read from 2,002 test samples,
            'all' will read from all 16,622 samples. Default: None, all samples.
        num_samples (int, optional): Number of samples (rows) to read. Default: None, reads the full dataset.
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> udpos_dataset_dir = "/path/to/udpos_dataset_dir"
        >>> dataset = ds.UDPOSDataset(dataset_dir=udpos_dataset_dir, usage='all')

    About UDPOS dataset:

    Text corpus dataset that clarifies syntactic or semantic sentence structure.
    The corpus comprises 254,830 words and 16,622 sentences, taken from various web media including
    weblogs, newsgroups, emails and reviews.

    Citation:

    .. code-block::

        @inproceedings{silveira14gold,
          year = {2014},
          author = {Natalia Silveira and Timothy Dozat and Marie-Catherine de Marneffe and Samuel Bowman
            and Miriam Connor and John Bauer and Christopher D. Manning},
          title = {A Gold Standard Dependency Corpus for {E}nglish},
          booktitle = {Proceedings of the Ninth International Conference on Language
            Resources and Evaluation (LREC-2014)}
        }
    """

    @check_udpos_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.UDPOSNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                             self.shard_id)


class WikiTextDataset(SourceDataset, TextBaseDataset):
    """
    WikiText2 and WikiText103 datasets.

    The generated dataset has one column :py:obj:`[text]` , and
    the tensor of column `text` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'test', 'valid' and 'all'. Default: None, all samples.
        num_samples (int, optional): Number of samples (rows) to read. Default: None, reads the full dataset.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files or invalid.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `num_samples` is invalid (< 0).
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    About WikiTextDataset dataset:

    The WikiText Long Term Dependency Language Modeling Dataset is an English lexicon containing 100 million words.
    These terms are drawn from Wikipedia's premium and benchmark articles, including versions of Wikitext2 and
    Wikitext103. For WikiText2, it has 36718 lines in wiki.train.tokens, 4358 lines in wiki.test.tokens and
    3760 lines in wiki.valid.tokens. For WikiText103, it has 1801350 lines in wiki.train.tokens, 4358 lines in
    wiki.test.tokens and 3760 lines in wiki.valid.tokens.

    Here is the original WikiText dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── WikiText2/WikiText103
             ├── wiki.train.tokens
             ├── wiki.test.tokens
             ├── wiki.valid.tokens

    Citation:

    .. code-block::

        @article{merity2016pointer,
          title={Pointer sentinel mixture models},
          author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
          journal={arXiv preprint arXiv:1609.07843},
          year={2016}
        }

    Examples:
        >>> wiki_text_dataset_dir = "/path/to/wiki_text_dataset_directory"
        >>> dataset = ds.WikiTextDataset(dataset_dir=wiki_text_dataset_dir, usage='all')
    """

    @check_wiki_text_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.WikiTextNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                self.shard_id)


class YahooAnswersDataset(SourceDataset, TextBaseDataset):
    """
    YahooAnswers dataset.

    The generated dataset has four columns :py:obj:`[class, title, content, answer]` , whose data type is string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'. 'train' will read
            from 1,400,000 train samples, 'test' will read from 60,000 test samples, 'all' will read from
            all 1,460,000 samples. Default: None, all samples.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will include all text.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> yahoo_answers_dataset_dir = "/path/to/yahoo_answers_dataset_directory"
        >>>
        >>> # 1) Read 3 samples from YahooAnswers dataset
        >>> dataset = ds.YahooAnswersDataset(dataset_dir=yahoo_answers_dataset_dir, num_samples=3)
        >>>
        >>> # 2) Read train samples from YahooAnswers dataset
        >>> dataset = ds.YahooAnswersDataset(dataset_dir=yahoo_answers_dataset_dir, usage="train")

    About YahooAnswers dataset:

    The YahooAnswers dataset consists of 630,000 text samples in 10 classes,
    There are 560,000 samples in the train.csv and 70,000 samples in the test.csv.
    The 10 different classes represent Society & Culture, Science & Mathematics, Health, Education & Reference,
    Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships,
    Politics & Government.

    Here is the original YahooAnswers dataset structure.
    You can unzip the dataset files into this directory structure and read by Mindspore's API.

    .. code-block::

        .
        └── yahoo_answers_dataset_dir
            ├── train.csv
            ├── test.csv
            ├── classes.txt
            └── readme.txt

    Citation:

    .. code-block::

        @article{YahooAnswers,
        title   = {Yahoo! Answers Topic Classification Dataset},
        author  = {Xiang Zhang},
        year    = {2015},
        howpublished = {}
        }
    """

    @check_yahoo_answers_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.YahooAnswersNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                                    self.num_shards, self.shard_id)


class YelpReviewDataset(SourceDataset, TextBaseDataset):
    """
    Yelp Review Polarity and Yelp Review Full datasets.

    The generated dataset has two columns: :py:obj:`[label, text]` , and the data type of two columns is string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'.
            For Polarity, 'train' will read from 560,000 train samples, 'test' will read from 38,000 test samples,
            'all' will read from all 598,000 samples.
            For Full, 'train' will read from 650,000 train samples, 'test' will read from 50,000 test samples,
            'all' will read from all 700,000 samples. Default: None, all samples.
        num_samples (int, optional): Number of samples (rows) to read. Default: None, reads all samples.
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> yelp_review_dataset_dir = "/path/to/yelp_review_dataset_dir"
        >>> dataset = ds.YelpReviewDataset(dataset_dir=yelp_review_dataset_dir, usage='all')

    About YelpReview Dataset:

    The Yelp Review Full dataset consists of reviews from Yelp. It is extracted from the Yelp Dataset Challenge 2015
    data, and it is mainly used for text classification.

    The Yelp Review Polarity dataset is constructed from the above dataset, by considering stars 1 and 2 negative, and 3
    and 4 positive.

    The directory structures of these two datasets are the same.
    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── yelp_review_dir
             ├── train.csv
             ├── test.csv
             └── readme.txt

    Citation:

    For Yelp Review Polarity:

    .. code-block::

        @article{zhangCharacterlevelConvolutionalNetworks2015,
          archivePrefix = {arXiv},
          eprinttype = {arxiv},
          eprint = {1509.01626},
          primaryClass = {cs},
          title = {Character-Level {{Convolutional Networks}} for {{Text Classification}}},
          abstract = {This article offers an empirical exploration on the use of character-level convolutional networks
                      (ConvNets) for text classification. We constructed several large-scale datasets to show that
                      character-level convolutional networks could achieve state-of-the-art or competitive results.
                      Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF
                      variants, and deep learning models such as word-based ConvNets and recurrent neural networks.},
          journal = {arXiv:1509.01626 [cs]},
          author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          month = sep,
          year = {2015},
        }

    Citation:

    For Yelp Review Full:

    .. code-block::

        @article{zhangCharacterlevelConvolutionalNetworks2015,
          archivePrefix = {arXiv},
          eprinttype = {arxiv},
          eprint = {1509.01626},
          primaryClass = {cs},
          title = {Character-Level {{Convolutional Networks}} for {{Text Classification}}},
          abstract = {This article offers an empirical exploration on the use of character-level convolutional networks
                      (ConvNets) for text classification. We constructed several large-scale datasets to show that
                      character-level convolutional networks could achieve state-of-the-art or competitive results.
                      Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF
                      variants, and deep learning models such as word-based ConvNets and recurrent neural networks.},
          journal = {arXiv:1509.01626 [cs]},
          author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          month = sep,
          year = {2015},
        }
    """

    @check_yelp_review_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.YelpReviewNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                                  self.num_shards, self.shard_id)
