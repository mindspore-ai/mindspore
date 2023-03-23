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
This file contains specific audio dataset loading classes. You can easily use
these classes to load the prepared dataset. For example:
    LJSpeechDataset: which is lj speech dataset.
    YesNoDataset: which is yes or no dataset.
    SpeechCommandsDataset: which is speech commands dataset.
    TedliumDataset: which is tedlium dataset.
    ...
After declaring the dataset object, you can further apply dataset operations
(e.g. filter, skip, concat, map, batch) on it.
"""
import mindspore._c_dataengine as cde

from .datasets import AudioBaseDataset, MappableDataset
from .validators import check_cmu_arctic_dataset, check_gtzan_dataset, check_libri_tts_dataset, \
    check_lj_speech_dataset, check_speech_commands_dataset, check_tedlium_dataset, check_yes_no_dataset

from ..core.validator_helpers import replace_none


class CMUArcticDataset(MappableDataset, AudioBaseDataset):
    """
    CMU Arctic dataset.

    The generated dataset has four columns: :py:obj:`[waveform, sample_rate, transcript, utterance_id]` .
    The tensor of column :py:obj:`waveform` is of the float32 type.
    The tensor of column :py:obj:`sample_rate` is of a scalar of uint32 type.
    The tensor of column :py:obj:`transcript` is of a scalar of string type.
    The tensor of column :py:obj:`utterance_id` is of a scalar of string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        name (str, optional): Part of this dataset, can be 'aew', 'ahw', 'aup', 'awb', 'axb', 'bdl',
            'clb', 'eey', 'fem', 'gka', 'jmk', 'ksp', 'ljm', 'lnh', 'rms', 'rxr', 'slp' or 'slt'.
            Default: None, means 'aew'.
        num_samples (int, optional): The number of audio to be included in the dataset.
            Default: None, will read all audio.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None, no dividing.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None, will use 0. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - Not support `mindspore.dataset.PKSampler` for `sampler` parameter yet.
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
        >>> cmu_arctic_dataset_directory = "/path/to/cmu_arctic_dataset_directory"
        >>>
        >>> # 1) Read 500 samples (audio files) in cmu_arctic_dataset_directory
        >>> dataset = ds.CMUArcticDataset(cmu_arctic_dataset_directory, name="ahw", num_samples=500)
        >>>
        >>> # 2) Read all samples (audio files) in cmu_arctic_dataset_directory
        >>> dataset = ds.CMUArcticDataset(cmu_arctic_dataset_directory)

    About CMUArctic dataset:

    The CMU Arctic databases are designed for the purpose of speech synthesis research.
    These single speaker speech databases have been carefully recorded under studio conditions
    and consist of approximately 1200 phonetically balanced English utterances. In addition to wavefiles,
    the databases provide complete support for the Festival Speech Synthesis System, including pre-built
    voices that may be used as is. The entire package is distributed as free software, without restriction
    on commercial or non-commercial use.

    You can construct the following directory structure from CMUArctic dataset and read by MindSpore's API.

    .. code-block::

        .
        └── cmu_arctic_dataset_directory
            ├── cmu_us_aew_arctic
            │    ├── wav
            │    │    ├──arctic_a0001.wav
            │    │    ├──arctic_a0002.wav
            │    │    ├──...
            │    ├── etc
            │    │    └── txt.done.data
            ├── cmu_us_ahw_arctic
            │    ├── wav
            │    │    ├──arctic_a0001.wav
            │    │    ├──arctic_a0002.wav
            │    │    ├──...
            │    └── etc
            │         └── txt.done.data
            └──...

    Citation:

    .. code-block::

        @article{LTI2003CMUArctic,
        title        = {CMU ARCTIC databases for speech synthesis},
        author       = {John Kominek and Alan W Black},
        journal      = {Language Technologies Institute [Online]},
        year         = {2003}
        howpublished = {http://www.festvox.org/cmu_arctic/}
        }
    """

    @check_cmu_arctic_dataset
    def __init__(self, dataset_dir, name=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.name = replace_none(name, "aew")

    def parse(self, children=None):
        return cde.CMUArcticNode(self.dataset_dir, self.name, self.sampler)


class GTZANDataset(MappableDataset, AudioBaseDataset):
    """
    GTZAN dataset.

    The generated dataset has three columns: :py:obj:`[waveform, sample_rate, label]` .
    The tensor of column :py:obj:`waveform` is of the float32 type.
    The tensor of column :py:obj:`sample_rate` is of a scalar of uint32 type.
    The tensor of column :py:obj:`label` is of a scalar of string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'valid', 'test' or 'all'.
            Default: None, will read all samples.
        num_samples (int, optional): The number of audio to be included in the dataset.
            Default: None, will read all audio.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - Not support `mindspore.dataset.PKSampler` for `sampler` parameter yet.
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
        >>> gtzan_dataset_directory = "/path/to/gtzan_dataset_directory"
        >>>
        >>> # 1) Read 500 samples (audio files) in gtzan_dataset_directory
        >>> dataset = ds.GTZANDataset(gtzan_dataset_directory, usage="all", num_samples=500)
        >>>
        >>> # 2) Read all samples (audio files) in gtzan_dataset_directory
        >>> dataset = ds.GTZANDataset(gtzan_dataset_directory)

    About GTZAN dataset:

    The GTZAN dataset appears in at least 100 published works and is the most commonly used
    public dataset for evaluation in machine listening research for music genre recognition.
    It consists of 1000 audio tracks, each of which is 30 seconds long. It contains 10 genres (blues,
    classical, country, disco, hiphop, jazz, metal, pop, reggae and rock), each of which is
    represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

    You can construct the following directory structure from GTZAN dataset and read by MindSpore's API.

    .. code-block::

        .
        └── gtzan_dataset_directory
            ├── blues
            │    ├──blues.00000.wav
            │    ├──blues.00001.wav
            │    ├──blues.00002.wav
            │    ├──...
            ├── disco
            │    ├──disco.00000.wav
            │    ├──disco.00001.wav
            │    ├──disco.00002.wav
            │    └──...
            └──...

    Citation:

    .. code-block::

        @misc{tzanetakis_essl_cook_2001,
        author    = "Tzanetakis, George and Essl, Georg and Cook, Perry",
        title     = "Automatic Musical Genre Classification Of Audio Signals",
        url       = "http://ismir2001.ismir.net/pdf/tzanetakis.pdf",
        publisher = "The International Society for Music Information Retrieval",
        year      = "2001"
        }
    """

    @check_gtzan_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.GTZANNode(self.dataset_dir, self.usage, self.sampler)


class LibriTTSDataset(MappableDataset, AudioBaseDataset):
    """
    LibriTTS dataset.

    The generated dataset has seven columns :py:obj:`[waveform, sample_rate, original_text, normalized_text,
    speaker_id, chapter_id, utterance_id]` .
    The tensor of column :py:obj:`waveform` is of the float32 type.
    The tensor of column :py:obj:`sample_rate` is of a scalar of uint32 type.
    The tensor of column :py:obj:`original_text` is of a scalar of string type.
    The tensor of column :py:obj:`normalized_text` is of a scalar of string type.
    The tensor of column :py:obj:`speaker_id` is of a scalar of uint32 type.
    The tensor of column :py:obj:`chapter_id` is of a scalar of uint32 type.
    The tensor of column :py:obj:`utterance_id` is of a scalar of string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Part of this dataset, can be 'dev-clean', 'dev-other', 'test-clean', 'test-other',
            'train-clean-100', 'train-clean-360', 'train-other-500', or 'all'. Default: None, means 'all'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all audio.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - Not support `mindspore.dataset.PKSampler` for `sampler` parameter yet.
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
        >>> libri_tts_dataset_dir = "/path/to/libri_tts_dataset_directory"
        >>>
        >>> # 1) Read 500 samples (audio files) in libri_tts_dataset_directory
        >>> dataset = ds.LibriTTSDataset(libri_tts_dataset_dir, usage="train-clean-100", num_samples=500)
        >>>
        >>> # 2) Read all samples (audio files) in libri_tts_dataset_directory
        >>> dataset = ds.LibriTTSDataset(libri_tts_dataset_dir)

    About LibriTTS dataset:

    LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz
    sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members.
    The LibriTTS corpus is designed for TTS research. It is derived from the original materials (mp3 audio
    files from LibriVox and text files from Project Gutenberg) of the LibriSpeech corpus.

    You can construct the following directory structure from LibriTTS dataset and read by MindSpore's API.

    .. code-block::

        .
        └── libri_tts_dataset_directory
            ├── dev-clean
            │    ├── 116
            │    │    ├── 288045
            |    |    |    ├── 116_288045.trans.tsv
            │    │    │    ├── 116_288045_000003_000000.wav
            │    │    │    └──...
            │    │    ├── 288046
            |    |    |    ├── 116_288046.trans.tsv
            |    |    |    ├── 116_288046_000003_000000.wav
            │    |    |    └── ...
            |    |    └── ...
            │    ├── 1255
            │    │    ├── 138279
            |    |    |    ├── 1255_138279.trans.tsv
            │    │    │    ├── 1255_138279_000001_000000.wav
            │    │    │    └── ...
            │    │    ├── 74899
            |    |    |    ├── 1255_74899.trans.tsv
            |    |    |    ├── 1255_74899_000001_000000.wav
            │    |    |    └── ...
            |    |    └── ...
            |    └── ...
            └── ...

    Citation:

    .. code-block::

        @article{lecun2010mnist,
        title        = {LIBRITTS handwritten digit database},
        author       = {zpw, NBU},
        journal      = {ATT Labs [Online]},
        volume       = {2},
        year         = {2010},
        howpublished = {http://www.openslr.org/resources/60/},
        description  = {The LibriSpeech ASR corpus (http://www.openslr.org/12/) [1] has been used in
                        various research projects. However, as it was originally designed for ASR research,
                        there are some undesired properties when using for TTS research}
        }
    """

    @check_libri_tts_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.LibriTTSNode(self.dataset_dir, self.usage, self.sampler)


class LJSpeechDataset(MappableDataset, AudioBaseDataset):
    """
    LJSpeech dataset.

    The generated dataset has four columns :py:obj:`[waveform, sample_rate, transcription, normalized_transcript]` .
    The column :py:obj:`waveform` is a tensor of the float32 type.
    The column :py:obj:`sample_rate` is a scalar of the int32 type.
    The column :py:obj:`transcription` is a scalar of the string type.
    The column :py:obj:`normalized_transcript` is a scalar of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of audios to be included in the dataset.
            Default: None, all audios.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into.
            Default: None. When this argument is specified, `num_samples` reflects
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
        >>> lj_speech_dataset_dir = "/path/to/lj_speech_dataset_directory"
        >>>
        >>> # 1) Get all samples from LJSPEECH dataset in sequence
        >>> dataset = ds.LJSpeechDataset(dataset_dir=lj_speech_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from LJSPEECH dataset
        >>> dataset = ds.LJSpeechDataset(dataset_dir=lj_speech_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from LJSPEECH dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.LJSpeechDataset(dataset_dir=lj_speech_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In LJSPEECH dataset, each dictionary has keys "waveform", "sample_rate", "transcription"
        >>> # and "normalized_transcript"

    About LJSPEECH dataset:

    This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker
    reading passages from 7 non-fiction books. A transcription is provided for each clip.
    Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.

    The texts were published between 1884 and 1964, and are in the public domain.
    The audio was recorded in 2016-17 by the LibriVox project and is also in the public domain.

    Here is the original LJSPEECH dataset structure.
    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── LJSpeech-1.1
            ├── README
            ├── metadata.csv
            └── wavs
                ├── LJ001-0001.wav
                ├── LJ001-0002.wav
                ├── LJ001-0003.wav
                ├── LJ001-0004.wav
                ├── LJ001-0005.wav
                ├── LJ001-0006.wav
                ├── LJ001-0007.wav
                ├── LJ001-0008.wav
                ...
                ├── LJ050-0277.wav
                └── LJ050-0278.wav

    Citation:

    .. code-block::

        @misc{lj_speech17,
        author       = {Keith Ito and Linda Johnson},
        title        = {The LJ Speech Dataset},
        howpublished = {url{https://keithito.com/LJ-Speech-Dataset}},
        year         = 2017
        }
    """

    @check_lj_speech_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.LJSpeechNode(self.dataset_dir, self.sampler)


class SpeechCommandsDataset(MappableDataset, AudioBaseDataset):
    """
    Speech Commands dataset.

    The generated dataset has five columns :py:obj:`[waveform, sample_rate, label, speaker_id, utterance_number]` .
    The tensor of column :py:obj:`waveform` is a vector of the float32 type.
    The tensor of column :py:obj:`sample_rate` is a scalar of the int32 type.
    The tensor of column :py:obj:`label` is a scalar of the string type.
    The tensor of column :py:obj:`speaker_id` is a scalar of the string type.
    The tensor of column :py:obj:`utterance_number` is a scalar of the int32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test', 'valid' or 'all'. 'train'
            will read from 84,843 samples, 'test' will read from 11,005 samples, 'valid' will read from 9,981
            test samples and 'all' will read from all 105,829 samples. Default: None, will read all samples.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will read all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument can only be specified
            when `num_shards` is also specified.
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
        >>> speech_commands_dataset_dir = "/path/to/speech_commands_dataset_directory"
        >>>
        >>> # Read 3 samples from SpeechCommands dataset
        >>> dataset = ds.SpeechCommandsDataset(dataset_dir=speech_commands_dataset_dir, num_samples=3)

    About SpeechCommands dataset:

    The SpeechCommands is database for limited_vocabulary speech recognition, containing 105,829 audio samples of
    '.wav' format.

    Here is the original SpeechCommands dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── speech_commands_dataset_dir
             ├── cat
                  ├── b433eff_nohash_0.wav
                  ├── 5a33edf_nohash_1.wav
                  └──....
             ├── dog
                  ├── b433w2w_nohash_0.wav
                  └──....
             ├── four
             └── ....

    Citation:

    .. code-block::

        @article{2018Speech,
        title={Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
        author={Warden, P.},
        year={2018}
        }
    """

    @check_speech_commands_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.SpeechCommandsNode(self.dataset_dir, self.usage, self.sampler)


class TedliumDataset(MappableDataset, AudioBaseDataset):
    """
    Tedlium dataset.
    The columns of generated dataset depend on the source SPH files and the corresponding STM files.

    The generated dataset has six columns :py:obj:`[waveform, sample_rate, transcript, talk_id, speaker_id,
    identifier]` .

    The data type of column `waveform` is float32, the data type of column `sample_rate` is int32,
    and the data type of columns `transcript` , `talk_id` , `speaker_id` and `identifier` is string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        release (str): Release of the dataset, can be 'release1', 'release2', 'release3'.
        usage (str, optional): Usage of this dataset.
            For release1 or release2, can be 'train', 'test', 'dev' or 'all'.
            'train' will read from train samples,
            'test' will read from test samples,
            'dev' will read from dev samples,
            'all' will read from all samples.
            For release3, can only be 'all', it will read from data samples. Default: None, all samples.
        extensions (str, optional): Extensions of the SPH files, only '.sph' is valid.
            Default: None, ".sph".
        num_samples (int, optional): The number of audio samples to be included in the dataset.
            Default: None, all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain stm files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
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
        >>> # 1) Get all train samples from TEDLIUM_release1 dataset in sequence.
        >>> dataset = ds.TedliumDataset(dataset_dir="/path/to/tedlium1_dataset_directory",
        ...                             release="release1", shuffle=False)
        >>>
        >>> # 2) Randomly select 10 samples from TEDLIUM_release2 dataset.
        >>> dataset = ds.TedliumDataset(dataset_dir="/path/to/tedlium2_dataset_directory",
        ...                             release="release2", num_samples=10, shuffle=True)
        >>>
        >>> # 3) Get samples from TEDLIUM_release-3 dataset for shard 0 in a 2-way distributed training.
        >>> dataset = ds.TedliumDataset(dataset_dir="/path/to/tedlium3_dataset_directory",
        ...                             release="release3", num_shards=2, shard_id=0)
        >>>
        >>> # In TEDLIUM dataset, each dictionary has keys : waveform, sample_rate, transcript, talk_id,
        >>> # speaker_id and identifier.

    About TEDLIUM_release1 dataset:

    The TED-LIUM corpus is English-language TED talks, with transcriptions, sampled at 16kHz.
    It contains about 118 hours of speech.

    About TEDLIUM_release2 dataset:

    This is the TED-LIUM corpus release 2, licensed under Creative Commons BY-NC-ND 3.0. All talks and text are
    property of TED Conferences LLC. The TED-LIUM corpus was made from audio talks and their transcriptions available
    on the TED website. We have prepared and filtered these data in order to train acoustic models to participate to
    the International Workshop on Spoken Language Translation 2011 (the LIUM English/French SLT system reached the
    first rank in the SLT task).

    About TEDLIUM_release-3 dataset:

    This is the TED-LIUM corpus release 3, licensed under Creative Commons BY-NC-ND 3.0. All talks and text are
    property of TED Conferences LLC. This new TED-LIUM release was made through a collaboration between the Ubiqus
    company and the LIUM (University of Le Mans, France).

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    The structure of TEDLIUM release2 is the same as TEDLIUM release1, only the data is different.

    .. code-block::

        .
        └──TEDLIUM_release1
            └── dev
                ├── sph
                    ├── AlGore_2009.sph
                    ├── BarrySchwartz_2005G.sph
                ├── stm
                    ├── AlGore_2009.stm
                    ├── BarrySchwartz_2005G.stm
            └── test
                ├── sph
                    ├── AimeeMullins_2009P.sph
                    ├── BillGates_2010.sph
                ├── stm
                    ├── AimeeMullins_2009P.stm
                    ├── BillGates_2010.stm
            └── train
                ├── sph
                    ├── AaronHuey_2010X.sph
                    ├── AdamGrosser_2007.sph
                ├── stm
                    ├── AaronHuey_2010X.stm
                    ├── AdamGrosser_2007.stm
            └── readme
            └── TEDLIUM.150k.dic

    The directory structure of TEDLIUM release3 is slightly different.

    .. code-block::

        .
        └──TEDLIUM_release-3
            └── data
                ├── ctl
                ├── sph
                    ├── 911Mothers_2010W.sph
                    ├── AalaElKhani.sph
                ├── stm
                    ├── 911Mothers_2010W.stm
                    ├── AalaElKhani.stm
            └── doc
            └── legacy
            └── LM
            └── speaker-adaptation
            └── readme
            └── TEDLIUM.150k.dic

    Citation:

    .. code-block::

        @article{
          title={TED-LIUM: an automatic speech recognition dedicated corpus},
          author={A. Rousseau, P. Deléglise, Y. Estève},
          journal={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
          year={May 2012},
          biburl={https://www.openslr.org/7/}
        }

        @article{
          title={Enhancing the TED-LIUM Corpus with Selected Data for Language Modeling and More TED Talks},
          author={A. Rousseau, P. Deléglise, and Y. Estève},
          journal={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
          year={May 2014},
          biburl={https://www.openslr.org/19/}
        }

        @article{
          title={TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation},
          author={François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Estève},
          journal={the 20th International Conference on Speech and Computer (SPECOM 2018)},
          year={September 2018},
          biburl={https://www.openslr.org/51/}
        }
    """

    @check_tedlium_dataset
    def __init__(self, dataset_dir, release, usage=None, extensions=None, num_samples=None,
                 num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.extensions = replace_none(extensions, ".sph")
        self.release = release
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.TedliumNode(self.dataset_dir, self.release, self.usage, self.extensions, self.sampler)


class YesNoDataset(MappableDataset, AudioBaseDataset):
    """
    YesNo dataset.

    The generated dataset has three columns :py:obj:`[waveform, sample_rate, labels]` .
    The tensor of column :py:obj:`waveform` is a vector of the float32 type.
    The tensor of column :py:obj:`sample_rate` is a scalar of the int32 type.
    The tensor of column :py:obj:`labels` is a scalar of the int32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: None, will use global default workers(8), it can be set
            by `mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument can only
            be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
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
        >>> yes_no_dataset_dir = "/path/to/yes_no_dataset_directory"
        >>>
        >>> # Read 3 samples from YesNo dataset
        >>> dataset = ds.YesNoDataset(dataset_dir=yes_no_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In YesNo dataset, each dictionary has keys "waveform", "sample_rate", "label"

    About YesNo dataset:

    Yesno is an audio dataset consisting of 60 recordings of one individual saying yes or no in Hebrew; each
    recording is eight words long.

    Here is the original YesNo dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── yes_no_dataset_dir
             ├── 1_1_0_0_1_1_0_0.wav
             ├── 1_0_0_0_1_1_0_0.wav
             ├── 1_1_0_0_1_1_0_0.wav
             └──....

    Citation:

    .. code-block::

        @NetworkResource{Kaldi_audio_project,
        author    = {anonymous},
        url       = "http://wwww.openslr.org/1/"
        }
    """

    @check_yes_no_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.YesNoNode(self.dataset_dir, self.sampler)
