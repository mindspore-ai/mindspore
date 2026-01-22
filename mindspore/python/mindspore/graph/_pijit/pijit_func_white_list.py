# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
# ============================================================================
"""The module of parser python object, called by c++."""

import collections
import types
import math
import os
import numpy
from mindspore.nn import GraphCell, Cell
from mindspore.ops.primitive import Primitive, constexpr, _primexpr
from mindspore.ops.composite.base import GradOperation, _Grad
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.graph.api import jit
from mindspore.common.tensor import Tensor
from mindspore.common._register_for_tensor import Registry
from mindspore._c_expression import MetaFuncGraph_, function_id
from mindspore._c_expression import TensorPy as Tensor_
from mindspore.graph._parse.resources import convert_object_map
from mindspore import _checkparam as validator
from mindspore import Parameter, ParameterTuple
from mindspore.common.initializer import Zero
from mindspore.ops.function import array_func
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore._c_expression.np_dtypes import np_dtype_valid
from mindspore.common.dtype import type_size_in_bytes
from mindspore.communication._comm_helper import _is_initialized, _get_rank_helper, _get_local_rank_helper, \
    _get_size_helper, _get_local_size_helper, _get_world_rank_from_group_rank_helper, _get_group_ranks, \
    _get_group_rank_from_world_rank_helper, _set_elegant_exit_handle
from mindspore import SummaryCollector
from mindspore.train import ModelCheckpoint, LossMonitor
from mindspore.train.model import _FrameworkProfilerCallback
from mindspore.train.data_sink import _init_sink_dataset
from mindspore.train.summary import SummaryRecord
from mindspore.train._utils import _exec_datagraph
from mindspore.train.summary.writer import BaseWriter
from mindspore.train.serialization import _exec_save, load, export_split_mindir, _parse_ckpt_proto, \
    _generate_front_info_for_param_data_file, _get_data_file, _encrypt_data, _split_save, _save_mindir_together, \
    _load_into_param_dict
from mindspore.parallel import _cost_model_context
from mindspore.parallel._utils import _is_in_data_parallel_mode
from mindspore.run_check._check_version import check_version_and_env_config
from mindspore.dataset.callback.ds_callback import DSCallback, WaitedDSCallback
from mindspore.dataset.transforms.transforms import TensorOperation, Compose, Concatenate, Duplicate, Fill, Mask, \
    OneHot, PadEnd, Plugin, RandomApply, RandomChoice, Slice, TypeCast, Unique
from mindspore.dataset.text.transforms import AddToken, JiebaTokenizer, Lookup, Ngram, SentencePieceTokenizer, \
    SlidingWindow, ToNumber, ToVectors, Truncate, TruncateSequencePair, UnicodeCharTokenizer, WordpieceTokenizer, \
    BasicTokenizer, BertTokenizer, CaseFold, FilterWikipediaXML, NormalizeUTF8, RegexReplace, RegexTokenizer, \
    UnicodeScriptTokenizer, WhitespaceTokenizer
from mindspore.dataset.core.datatypes import nptype_to_detype, mstype_to_detype, mstypelist_to_detypelist
from mindspore.dataset.audio.utils import create_dct, linear_fbanks, melscale_fbanks
from mindspore.dataset.audio.transforms import AllpassBiquad, AmplitudeToDB, Angle, BandBiquad, BandpassBiquad, \
    BandrejectBiquad, BassBiquad, Biquad, ComplexNorm, ComputeDeltas, Contrast, DBToAmplitude, DCShift, DeemphBiquad, \
    DetectPitchFrequency, Dither, EqualizerBiquad, Fade, Filtfilt, Flanger, FrequencyMasking, Gain, GriffinLim, \
    HighpassBiquad, InverseMelScale, InverseSpectrogram, LFCC, LFilter, LowpassBiquad, Magphase, MaskAlongAxis, \
    MaskAlongAxisIID, MelScale, MelSpectrogram, MFCC, MuLawDecoding, MuLawEncoding, Overdrive, Phaser, PhaseVocoder, \
    PitchShift, Resample, RiaaBiquad, SlidingWindowCmn, SpectralCentroid, Spectrogram, TimeMasking, TimeStretch, \
    TrebleBiquad, Vad, Vol
from mindspore.dataset.engine.datasets_audio import CMUArcticDataset, GTZANDataset, LibriTTSDataset, LJSpeechDataset, \
    SpeechCommandsDataset, TedliumDataset, YesNoDataset
from mindspore.dataset.engine.cache_client import DatasetCache
from mindspore.dataset.engine.iterators import Iterator
from mindspore.dataset.engine.datasets_standard_format import CSVDataset, MindDataset, TFRecordDataset
from mindspore.dataset.engine.datasets_text import AGNewsDataset, AmazonReviewDataset, CLUEDataset, CoNLL2000Dataset, \
    DBpediaDataset, EnWik9Dataset, IMDBDataset, IWSLT2016Dataset, IWSLT2017Dataset, Multi30kDataset, WikiTextDataset, \
    PennTreebankDataset, SogouNewsDataset, SQuADDataset, SST2Dataset, TextFileDataset, UDPOSDataset, \
    YelpReviewDataset, YahooAnswersDataset
from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset
from mindspore.dataset.engine.datasets_vision import Caltech256Dataset, CelebADataset, Cifar100Dataset, \
    Cifar10Dataset, CityscapesDataset, CocoDataset, DIV2KDataset, EMnistDataset, FakeImageDataset, \
    FashionMnistDataset, FlickrDataset, Food101Dataset, ImageFolderDataset, KITTIDataset, KMnistDataset, \
    LFWDataset, LSUNDataset, ManifestDataset, MnistDataset, OmniglotDataset, PhotoTourDataset, \
    Places365Dataset, QMnistDataset, RandomDataset, RenderedSST2Dataset, SBUDataset, SemeionDataset, \
    STL10Dataset, SUN397Dataset, USPSDataset, VOCDataset, WIDERFaceDataset
from mindspore.dataset.engine.queue import _SharedQueue
from mindspore.dataset.engine.datasets import Dataset, BucketBatchByLengthDataset, BatchDataset, \
    PaddedBatchDataset, SyncWaitDataset, ShuffleDataset, MapDataset, FilterDataset, RepeatDataset, SkipDataset, \
    TakeDataset, ZipDataset, ConcatDataset, RenameDataset, ProjectDataset, _ToDevice, TransferDataset, Schema
from mindspore.dataset.engine.samplers import Sampler, DistributedSampler, PKSampler, RandomSampler, SubsetSampler, \
    SequentialSampler, SubsetRandomSampler, WeightedRandomSampler
from mindspore.dataset.vision.utils import encode_jpeg, encode_png, get_image_num_channels, get_image_size, \
    read_file, read_image, read_video, read_video_timestamps, write_file, write_jpeg, write_png
from mindspore.dataset.vision.transforms import AdjustBrightness, AdjustContrast, AdjustGamma as VAdjustGamma, \
    AdjustHue, AdjustSaturation, AdjustSharpness, Affine, AutoAugment as VAutoAugment, AutoContrast as VAutoContrast, \
    BoundingBoxAugment as VBoundingBoxAugment, CenterCrop as VCenterCrop, ConvertColor as VConvertColor, \
    Crop as VCrop, CutMixBatch as VCutMixBatch, CutOut as VCutOut, Decode as VDecode, DecodeVideo, \
    Equalize as VEqualize, Erase, GaussianBlur as VGaussianBlur, HorizontalFlip as VHorizontalFlip, \
    HWC2CHW as VHWC2CHW, Invert as VInvert, MixUpBatch as VMixUpBatch, Normalize as VNormalize, \
    NormalizePad as VNormalizePad, Pad as VPad, PadToSize, Perspective, Posterize, RandAugment, \
    RandomAdjustSharpness as VRandomAdjustSharpness, RandomAffine as VRandomAffine, \
    RandomAutoContrast as VRandomAutoContrast, RandomColor as VRandomColor, RandomColorAdjust as VRandomColorAdjust, \
    RandomCrop as VRandomCrop, RandomCropDecodeResize as VRandomCropDecodeResize, \
    RandomCropWithBBox as VRandomCropWithBBox, RandomEqualize as VRandomEqualize, RandomResize as VRandomResize, \
    RandomHorizontalFlip as VRandomHorizontalFlip, RandomHorizontalFlipWithBBox as VRandomHorizontalFlipWithBBox, \
    RandomInvert as VRandomInvert, RandomLighting as VRandomLighting, RandomPosterize as VRandomPosterize, \
    RandomResizedCrop as VRandomResizedCrop, RandomResizedCropWithBBox as VRandomResizedCropWithBBox, \
    RandomResizeWithBBox as VRandomResizeWithBBox, RandomRotation as VRandomRotation, \
    RandomSelectSubpolicy as VRandomSelectSubpolicy, RandomSharpness as VRandomSharpness, \
    RandomSolarize as VRandomSolarize, RandomVerticalFlip as VRandomVerticalFlip, \
    RandomVerticalFlipWithBBox as VRandomVerticalFlipWithBBox, Rescale as VRescale, Resize as VResize, ResizedCrop, \
    ResizeWithBBox as VResizeWithBBox, Rotate as VRotate, SlicePatches as VSlicePatches, Solarize, ToTensor,\
    TrivialAugmentWide, UniformAugment as VUniformAugment, VerticalFlip as VVerticalFlip
from mindspore.profiler.profiler import Profiler
from mindspore.communication._comm_helper import _create_group_helper, _destroy_group_helper
from mindspore.communication.management import _set_rank_from_mpi, init as cinit, release as crelease
from mindspore.hal.stream import Stream, synchronize, set_cur_stream, current_stream, default_stream
from mindspore.hal.event import Event
from mindspore.hal.memory import memory_stats, memory_reserved, max_memory_allocated, reset_peak_memory_stats, \
    memory_summary, memory_allocated, max_memory_reserved, reset_max_memory_allocated, reset_max_memory_reserved
from mindspore.multiprocessing import Process
from mindspore.mindrecord.config import encrypt, decrypt
from mindspore.mindrecord import FileWriter
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import MindPage
from mindspore.parallel._ps_context import ps_context
from mindspore.parallel.algo_parameter_config import _AlgoParameterConfig
from mindspore.parallel._utils import _reset_op_id
from mindspore.parallel._auto_parallel_context import _AutoParallelContext
from mindspore.graph.api import ms_memory_recycle
from mindspore.context import _Context


def _get_after_grad_code():
    """Get the code object of 'after_grad'"""
    name = "after_grad"
    codes = []
    for cnst in GradOperation.__call__.__code__.co_consts:
        if isinstance(cnst, types.CodeType) and cnst.co_name == name:
            codes.append(cnst)
    for cnst in _Grad.__call__.__code__.co_consts:
        if isinstance(cnst, types.CodeType) and cnst.co_name == name:
            codes.append(cnst)
    if not codes:
        raise RuntimeError("check GradOperation, can't find the code of 'after_grad'")
    return codes


def _get_dataset_forbidden_code():
    """Get the forbidden function which should be broken in graph"""
    codes = []
    codes.extend([DSCallback.__init__, WaitedDSCallback.__init__])
    codes.extend([TensorOperation.__call__, Compose.parse, Concatenate.__init__, Concatenate.parse, Duplicate.parse, \
                  Fill.__init__, Fill.parse, Mask.__init__, Mask.parse, OneHot.parse, PadEnd.__init__, PadEnd.parse, \
                  Plugin.parse, RandomApply.parse, RandomChoice.parse, Slice.parse, TypeCast.parse, Unique.parse])
    codes.extend([AddToken.parse, JiebaTokenizer.parse, Lookup.parse, Ngram.parse, SentencePieceTokenizer.parse, \
                  SlidingWindow.parse, ToNumber.parse, ToVectors.parse, Truncate.parse, TruncateSequencePair.parse, \
                  UnicodeCharTokenizer.parse, WordpieceTokenizer.parse, BasicTokenizer.parse, BertTokenizer.parse, \
                  CaseFold.parse, FilterWikipediaXML.parse, NormalizeUTF8.parse, RegexReplace.parse, \
                  RegexTokenizer.parse, UnicodeScriptTokenizer.parse, WhitespaceTokenizer.parse])
    codes.extend([create_dct, linear_fbanks, melscale_fbanks])
    codes.extend([AllpassBiquad.parse, AmplitudeToDB.parse, Angle.parse, BandBiquad.parse, BandpassBiquad.parse, \
                  BandrejectBiquad.parse, BassBiquad.parse, Biquad.parse, ComplexNorm.parse, ComputeDeltas.parse, \
                  Contrast.parse, DBToAmplitude.parse, DCShift.parse, DeemphBiquad.parse, DetectPitchFrequency.parse, \
                  Dither.parse, EqualizerBiquad.parse, Fade.parse, Filtfilt.parse, Flanger.parse, \
                  FrequencyMasking.parse, Gain.parse, GriffinLim.parse, HighpassBiquad.parse, InverseMelScale.parse, \
                  InverseSpectrogram.parse, LFCC.parse, LFilter.parse, LowpassBiquad.parse, Magphase.parse, \
                  MaskAlongAxis.parse, MaskAlongAxisIID.parse, MelScale.parse, MelSpectrogram.parse, MFCC.parse, \
                  MuLawDecoding.parse, MuLawEncoding.parse, Overdrive.parse, Phaser.parse, PhaseVocoder.parse, \
                  PhaseVocoder.__init__, PitchShift.parse, Resample.parse, RiaaBiquad.parse, SlidingWindowCmn.parse, \
                  SpectralCentroid.parse, Spectrogram.parse, TimeMasking.parse, TimeStretch.parse, \
                  TrebleBiquad.parse, Vad.parse, Vol.parse])
    codes.extend([CMUArcticDataset.parse, GTZANDataset.parse, LibriTTSDataset.parse, LJSpeechDataset.parse, \
                  SpeechCommandsDataset.parse, TedliumDataset.parse, YesNoDataset.parse])
    codes.extend([DatasetCache.__init__, Iterator.__init__])
    codes.extend([CSVDataset.parse, MindDataset.parse, TFRecordDataset.parse])
    codes.extend([AGNewsDataset.parse, AmazonReviewDataset.parse, CLUEDataset.parse, CoNLL2000Dataset.parse, \
                  DBpediaDataset.parse, EnWik9Dataset.parse, IMDBDataset.parse, IWSLT2016Dataset.parse, \
                  IWSLT2017Dataset.parse, Multi30kDataset.parse, PennTreebankDataset.parse, SogouNewsDataset.parse, \
                  SQuADDataset.parse, SST2Dataset.parse, TextFileDataset.parse, UDPOSDataset.parse, \
                  WikiTextDataset.parse, YahooAnswersDataset.parse, YelpReviewDataset.parse])
    codes.extend([GeneratorDataset.parse, Caltech256Dataset.parse, CelebADataset.parse, Cifar10Dataset.parse, \
                  Cifar100Dataset.parse, CityscapesDataset.parse, CocoDataset.parse, DIV2KDataset.parse, \
                  EMnistDataset.parse, FakeImageDataset.parse, FashionMnistDataset.parse, FlickrDataset.parse, \
                  Food101Dataset.parse, ImageFolderDataset.parse, KITTIDataset.parse, KMnistDataset.parse, \
                  LFWDataset.parse, LSUNDataset.parse, ManifestDataset.parse, MnistDataset.parse, VOCDataset.parse, \
                  OmniglotDataset.parse, PhotoTourDataset.parse, Places365Dataset.parse, QMnistDataset.parse, \
                  RandomDataset.parse, RenderedSST2Dataset.parse, SBUDataset.parse, SemeionDataset.parse, \
                  STL10Dataset.parse, SUN397Dataset.parse, USPSDataset.parse, WIDERFaceDataset.parse])
    codes.extend([_SharedQueue.put, _SharedQueue.get])
    codes.extend([BucketBatchByLengthDataset.parse, BatchDataset.parse, \
                  PaddedBatchDataset.parse, SyncWaitDataset.parse, ShuffleDataset.parse, MapDataset.parse, \
                  FilterDataset.parse, RepeatDataset.parse, SkipDataset.parse, TakeDataset.parse, ZipDataset.parse, \
                  ConcatDataset.parse, RenameDataset.parse, ProjectDataset.parse, _ToDevice.__init__, \
                  TransferDataset.parse, Schema.__init__, Schema.add_column, Dataset.save])
    codes.extend([Sampler.parse, DistributedSampler.parse, DistributedSampler.parse_for_minddataset, PKSampler.parse, \
                  PKSampler.parse_for_minddataset, RandomSampler.parse, RandomSampler.parse_for_minddataset, \
                  SequentialSampler.parse, SequentialSampler.parse_for_minddataset, SubsetSampler.parse, \
                  SubsetSampler.parse_for_minddataset, SubsetRandomSampler.parse, \
                  SubsetRandomSampler.parse_for_minddataset, WeightedRandomSampler.parse])
    codes.extend([encode_jpeg, encode_png, get_image_num_channels, get_image_size, read_file, read_image, read_video, \
                  read_video_timestamps, write_file, write_jpeg, write_png])
    codes.extend([AdjustBrightness.parse, AdjustContrast.parse, VAdjustGamma.parse, AdjustHue.parse, \
                  AdjustSaturation.parse, AdjustSharpness.parse, Affine.parse, VAutoAugment.parse, \
                  VAutoContrast.parse, VBoundingBoxAugment.parse, VCenterCrop.parse, VConvertColor.parse, \
                  VCrop.parse, VCutMixBatch.parse, VCutOut.parse, VDecode.parse, DecodeVideo.parse, \
                  VEqualize.parse, Erase.parse, VGaussianBlur.parse, VHorizontalFlip.parse, \
                  VHWC2CHW.parse, VInvert.parse, VMixUpBatch.parse, VNormalize.parse, VNormalizePad.parse, \
                  VPad.parse, PadToSize.parse, Perspective.parse, Posterize.parse, RandAugment.parse, \
                  VRandomAdjustSharpness.parse, VRandomAffine.parse, \
                  VRandomAutoContrast.parse, VRandomColor.parse, VRandomColorAdjust.parse, \
                  VRandomCrop.parse, VRandomCropDecodeResize.parse, \
                  VRandomCropWithBBox.parse, VRandomEqualize.parse, VRandomResize.parse, \
                  VRandomHorizontalFlip.parse, VRandomHorizontalFlipWithBBox.parse, \
                  VRandomInvert.parse, VRandomLighting.parse, VRandomPosterize.parse, \
                  VRandomResizedCrop.parse, VRandomResizedCropWithBBox.parse, \
                  VRandomResizeWithBBox.parse, VRandomRotation.parse, \
                  VRandomSelectSubpolicy.parse, VRandomSharpness.parse, \
                  VRandomSolarize.parse, VRandomVerticalFlip.parse, \
                  VRandomVerticalFlipWithBBox.parse, VRescale.parse, VResize.parse, ResizedCrop.parse, \
                  VResizeWithBBox.parse, VRotate.parse, VSlicePatches.parse, Solarize.parse, ToTensor.parse,\
                  TrivialAugmentWide.parse, VUniformAugment.parse, VVerticalFlip])
    return codes


def _get_math_code():
    """Get the math builtin function which should be guarded in graph"""
    codes = []
    for i in dir(math):
        codes.append(getattr(math, i))
    return codes


def _get_psjit_code():
    """Get the code object of 'staging_specialize'"""
    @jit
    def inner():
        pass
    return inner.__code__


def _get_constexpr_code():
    """Get the code object of '@constexpr'"""
    @constexpr
    def inner():
        pass
    code = inner.__call__.__code__
    # check it before c++ use it
    if not isinstance(inner, Primitive) or code is Primitive.__call__.__code__:
        raise RuntimeError("@constexpr not isinstance(inner, Primitive) or code is Primitive.__call__.__code__")
    return code


def _get_primexpr_code():
    """Get the code object of '@_primexpr'"""
    @_primexpr
    def inner():
        pass
    code = inner.__call__.__code__
    # check it before c++ use it
    if not isinstance(inner, Primitive) or code is Primitive.__call__.__code__:
        raise RuntimeError("@_primexpr not isinstance(inner, Primitive) or code is Primitive.__call__.__code__")
    return code


def _pijit_constexpr():
    """Placeholder for uniqure id"""


def _get_pijit_constexpr_code():
    codes = []
    for cnst in validator.check_transpose_axis.__code__.co_consts:
        if isinstance(cnst, types.CodeType) and cnst.co_name == "_check_dim":
            codes.append(cnst)
    return codes


psjit_code = _get_psjit_code()
constexpr_code = _get_constexpr_code()
primexpr_code = _get_primexpr_code()

primitive_key = id(Primitive.__call__)
primitive_assign_key = id(P.Assign.__call__)

constexpr_key = id(constexpr_code)
primexpr_key = id(primexpr_code)
meta_func_graph_key = id(MetaFuncGraph_)
pijit_forbidden_key = id(NotImplemented)
pijit_constexpr_key = id(_pijit_constexpr)


# check WrapperDescriptor: function_id(tuple.__getitem__) == function_id(tuple().__getitem__)
# check MethodDescriptor: function_id(list.__getitem__) == function_id(list().__getitem__)
# check instancemethod: function_id(Tensor_.from_numpy) == function_id(Tensor_(1).from_numpy)
# check cfunction filter: function_id(Tensor_.from_numpy) != function_id(Tensor_._is_test_stub)
# check function id: function_id(Tensor.astype) == function_id(Tensor(1).astype) == id(Tensor.astype)
# check user defined object id: function_id(Primitive) == function_id(Primitive) == id(Primitive)


FUNC_KEY_EMPTY = 0  # ""
FUNC_KEY_PIJIT_CONSTEXPR = 1  # "pijit.constexpr"
FUNC_KEY_PIJIT_FORBIDDEN = 2  # "pijit.forbidden"
FUNC_KEY_BUILTIN_FUNC = 3  # "builtin.func"
FUNC_KEY_LIST_APPEND = 4  # "list.append"
FUNC_KEY_DICT_POP = 5  # "dict.pop"
FUNC_KEY_PRIMITIVE = 6  # "mindspore._c_expression.Primitive_"
FUNC_KEY_META_FUNCG_RAPH = 7  # "mindspore._c_expression.MetaFuncGraph_"
FUNC_KEY_PSJIT_CODE = 8  # "mindspore.graph.api.jit.<locals>.staging_specialize"
FUNC_KEY_CONSTEXPR = 9  # "mindspore.ops.primitive.constexpr"
FUNC_KEY_PRIMEXPR = 10  # "mindspore.ops.primitive._primexpr"
FUNC_KEY_GET_CACHE_PRIM = 11  # "mindspore.ops._primitive_cache._get_cache_prim"
FUNC_KEY_REGISTRY_GET = 12  # "mindspore.common._register_for_tensor.Registry.get"
FUNC_KEY_TENSOR_ASTYPE = 13  # "mindspore.common.tensor.Tensor.astype"
FUNC_KEY_GRAD_OPERATIONS_CODE = 14 # "mindspore.ops.composite.base._Grad.__call__.<locals>.after_grad"
FUNC_KEY_PSJIT_CONVERTMAP = 15 # "mindspore.graph._parse.resources.convert_object_map"
FUNC_KEY_GRAPH_CELL = 16  # "mindspore.nn.cell.GraphCell"
FUNC_KEY_MS_API = 17  # mindspore common api
FUNC_KEY_MAPPING_GET = 18 # collections.abc.Mapping.get
FUNC_KEY_LIST_POP = 19  # list.pop
FUNC_KEY_LIST_REMOVE = 20  # list.remove
FUNC_KEY_LIST_REVERSE = 21  # list.reverse
FUNC_KEY_DICT_ITEMS = 22  # dict.items
FUNC_KEY_PRIMITIVE_ASSIGN = 23  # mindspore.ops.assign, Primitive("Assign")
FUNC_KEY_TENSOR_SETITEM = 24  # Tensor.__setitem__
FUNC_KEY_TENSOR_ASSIGN_VALUE = 25  # Tensor.assign_value
FUNC_KEY_TENSOR_IS_CONTIGUOUS = 26  # Tensor.is_contiguous

# Initialized only once. This map will initialize by c++ when start pijit.
# key is customer if fuzzy match. (Primitive, constexpr, primexpr, MetaFuncGraph)
# key is id of code for nest object. (jit.<locals>.staging_specialize, GradOperation.__call__.<locals>.after_grad)
# key is id of object for callalbe object.
# key is cfunction pointer for builtin_function or method. (isinstance, tuple.__getitem__, Tensor_.asnumpy)
_func_map = {
    # special function
    pijit_constexpr_key: FUNC_KEY_PIJIT_CONSTEXPR,
    id(getattr(array_func, "_get_max_type")): FUNC_KEY_PIJIT_CONSTEXPR,
    id(Cell.__getattr__): FUNC_KEY_PIJIT_CONSTEXPR,
    pijit_forbidden_key: FUNC_KEY_PIJIT_FORBIDDEN,
    primitive_key: FUNC_KEY_PRIMITIVE,
    constexpr_key: FUNC_KEY_CONSTEXPR,
    primexpr_key: FUNC_KEY_PRIMEXPR,
    meta_func_graph_key: FUNC_KEY_META_FUNCG_RAPH,
    function_id(GraphCell.__call__): FUNC_KEY_GRAPH_CELL,
    id(psjit_code): FUNC_KEY_PSJIT_CODE,
    function_id(_get_cache_prim): FUNC_KEY_GET_CACHE_PRIM,
    function_id(Registry.get): FUNC_KEY_REGISTRY_GET,

    # tensor side-effect
    primitive_assign_key: FUNC_KEY_PRIMITIVE_ASSIGN,
    function_id(F.assign): FUNC_KEY_PRIMITIVE_ASSIGN,
    function_id(Tensor.assign_value): FUNC_KEY_TENSOR_ASSIGN_VALUE,
    function_id(Tensor.__setitem__): FUNC_KEY_TENSOR_SETITEM,

    # Tensor method
    function_id(Tensor.astype): FUNC_KEY_TENSOR_ASTYPE,

    # types.BuiltinFunctionType
    function_id(isinstance): FUNC_KEY_BUILTIN_FUNC,
    function_id(issubclass): FUNC_KEY_BUILTIN_FUNC,
    function_id(len): FUNC_KEY_BUILTIN_FUNC,
    function_id(abs): FUNC_KEY_BUILTIN_FUNC,
    function_id(max): FUNC_KEY_BUILTIN_FUNC,
    function_id(min): FUNC_KEY_BUILTIN_FUNC,
    function_id(all): FUNC_KEY_BUILTIN_FUNC,
    function_id(any): FUNC_KEY_BUILTIN_FUNC,
    function_id(hash): FUNC_KEY_BUILTIN_FUNC,
    function_id(id): FUNC_KEY_BUILTIN_FUNC,
    function_id(ord): FUNC_KEY_BUILTIN_FUNC,
    function_id(callable): FUNC_KEY_BUILTIN_FUNC,
    function_id(getattr): FUNC_KEY_BUILTIN_FUNC,
    function_id(hasattr): FUNC_KEY_BUILTIN_FUNC,
    function_id(chr): FUNC_KEY_BUILTIN_FUNC,
    function_id(divmod): FUNC_KEY_BUILTIN_FUNC,
    function_id(repr): FUNC_KEY_BUILTIN_FUNC,
    function_id(type): FUNC_KEY_BUILTIN_FUNC,

    # types.MethodDescriptorType, types.WrapperDescriptorType
    function_id(tuple.__getitem__): FUNC_KEY_BUILTIN_FUNC,
    function_id(tuple.count): FUNC_KEY_BUILTIN_FUNC,
    function_id(tuple.index): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.__getitem__): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.copy): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.index): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.count): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.__contains__): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.__getitem__): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.get): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.keys): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.values): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.items): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.fromkeys): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.copy): FUNC_KEY_BUILTIN_FUNC,
    function_id(set.__contains__): FUNC_KEY_BUILTIN_FUNC,
    function_id(set.copy): FUNC_KEY_BUILTIN_FUNC,
    function_id(set.issubset): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.find): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.count): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.index): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.rfind): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.rindex): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.startswith): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.endswith): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isascii): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.islower): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isupper): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.istitle): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isspace): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isdecimal): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isdigit): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isnumeric): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isalpha): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isalnum): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isidentifier): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isprintable): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.replace): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.format): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.format_map): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.__format__): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.append): FUNC_KEY_LIST_APPEND,
    function_id(list.pop): FUNC_KEY_LIST_POP,
    function_id(list.remove): FUNC_KEY_LIST_REMOVE,
    function_id(list.reverse): FUNC_KEY_LIST_REVERSE,
    function_id(dict.pop): FUNC_KEY_DICT_POP,
    function_id(dict.items): FUNC_KEY_DICT_ITEMS,

    # instancemethod
    function_id(Tensor_._is_test_stub): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_.__str__): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_.__repr__): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_.convert_bytes_to_tensor): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.dim): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.from_numpy): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.getitem_index_info): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.get_bytes): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.is_init): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.is_contiguous): FUNC_KEY_TENSOR_IS_CONTIGUOUS,
    function_id(Tensor_.stride): FUNC_KEY_BUILTIN_FUNC,
    # Tensor_.asnumpy need real tensor value

    # other builtin function
    function_id(collections.abc.Mapping.get): FUNC_KEY_MAPPING_GET,
    function_id(numpy.isinf): FUNC_KEY_BUILTIN_FUNC,
    function_id(numpy.isnan): FUNC_KEY_BUILTIN_FUNC,
    function_id(numpy.abs): FUNC_KEY_BUILTIN_FUNC,
    function_id(numpy.log): FUNC_KEY_BUILTIN_FUNC,
    function_id(os.getenv): FUNC_KEY_BUILTIN_FUNC,

    # const function
    function_id(validator.check_number_range): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(validator.check_is_int): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(validator.check_is_number): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(validator.check_positive_int_sequence): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(np_dtype_valid): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(_is_initialized): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(_set_elegant_exit_handle): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(_cost_model_context.get_cost_model_context): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(Stream.__repr__): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(_is_in_data_parallel_mode): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(check_version_and_env_config): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(Tensor.tolist): FUNC_KEY_PIJIT_CONSTEXPR,

    # inner function
    function_id(type_size_in_bytes): FUNC_KEY_BUILTIN_FUNC,
    function_id(_get_rank_helper): FUNC_KEY_BUILTIN_FUNC,
    function_id(_get_local_rank_helper): FUNC_KEY_BUILTIN_FUNC,
    function_id(_get_size_helper): FUNC_KEY_BUILTIN_FUNC,
    function_id(_get_local_size_helper): FUNC_KEY_BUILTIN_FUNC,
    function_id(_get_world_rank_from_group_rank_helper): FUNC_KEY_BUILTIN_FUNC,
    function_id(_get_group_ranks): FUNC_KEY_BUILTIN_FUNC,
    function_id(_get_group_rank_from_world_rank_helper): FUNC_KEY_BUILTIN_FUNC,
    function_id(nptype_to_detype): FUNC_KEY_BUILTIN_FUNC,
    function_id(mstype_to_detype): FUNC_KEY_BUILTIN_FUNC,
    function_id(mstypelist_to_detypelist): FUNC_KEY_BUILTIN_FUNC,

    # no need to capture function in black list
    function_id(SummaryCollector.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(SummaryCollector.begin): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(SummaryCollector.step_end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(SummaryCollector.epoch_end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(SummaryCollector.end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(ModelCheckpoint.step_end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(ModelCheckpoint.end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(LossMonitor.step_end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(LossMonitor.on_train_epoch_end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(SummaryRecord.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_exec_datagraph): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(BaseWriter.writer): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_FrameworkProfilerCallback.step_begin): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_FrameworkProfilerCallback.step_end): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_init_sink_dataset): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_exec_save): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(load): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(export_split_mindir): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_parse_ckpt_proto): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_generate_front_info_for_param_data_file): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_get_data_file): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_encrypt_data): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_split_save): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_save_mindir_together): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_load_into_param_dict): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Profiler.start): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_create_group_helper): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_destroy_group_helper): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_set_rank_from_mpi): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(cinit): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(crelease): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Stream.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Stream.synchronize): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Stream.query): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Stream.__eq__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(synchronize): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(set_cur_stream): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(current_stream): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(default_stream): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Event.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Event.record): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Event.wait): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Event.synchronize): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Event.query): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Event.elapsed_time): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(memory_stats): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(memory_reserved): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(max_memory_reserved): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(reset_peak_memory_stats): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(reset_peak_memory_stats): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(memory_summary): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(memory_allocated): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(max_memory_allocated): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(reset_max_memory_reserved): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(reset_max_memory_allocated): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Process.run): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(Process.start): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(encrypt): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(decrypt): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(FileWriter.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(FileReader.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(MindPage.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(ps_context): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_AlgoParameterConfig.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_reset_op_id): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_AutoParallelContext.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(ms_memory_recycle): FUNC_KEY_PIJIT_FORBIDDEN,
    function_id(_Context.__init__): FUNC_KEY_PIJIT_FORBIDDEN,
}

for after_grad in _get_after_grad_code():
    _func_map[id(after_grad)] = FUNC_KEY_GRAD_OPERATIONS_CODE

for func in _get_dataset_forbidden_code():
    _func_map[function_id(func)] = FUNC_KEY_PIJIT_FORBIDDEN

for math_func in _get_math_code():
    _func_map[function_id(math_func)] = FUNC_KEY_BUILTIN_FUNC

for k, v in convert_object_map.items():
    key = id(k)
    if key not in _func_map and isinstance(v, Primitive):
        if key is print:
            continue
        _func_map[key] = FUNC_KEY_PSJIT_CONVERTMAP

for const_code in _get_pijit_constexpr_code():
    _func_map[id(const_code)] = FUNC_KEY_PIJIT_CONSTEXPR

GUARD_KEY_RELAX_FUNC = 1
_guard_func_map = {}


def infer_after_grad(after_grad_func, inputs, outputs):
    """ Infer after_grad """
    contents = {}
    for index, name in enumerate(after_grad_func.__code__.co_freevars):
        contents[name] = after_grad_func.__closure__[index].cell_contents
    grad = contents.get('grad_') if 'grad_' in contents else contents.get('self')
    grad_position = contents.get('grad_position')
    weights = contents.get('weights')

    def _validation_check():
        if not grad:
            raise ValueError("Failed to get grad")
        if grad.has_aux and len(inputs) <= 1:
            raise ValueError("Hax_aux must have more than one input")
        if grad_position is None and weights is None:
            raise ValueError("Ethither grad_position or weights must not be empty")
        if grad.return_ids and grad_position is None:
            raise ValueError("return_ids must have grad_position not empty")

    def _to_result(grads):
        if not grads:
            return None
        if len(grads) == 1:
            return grads[0]
        return tuple(grads)

    _validation_check()

    input_grads = []
    if isinstance(grad_position, tuple):
        for pos in grad_position:
            input_grads.append((pos, inputs[pos]) if grad.return_ids else inputs[pos])
    elif isinstance(grad_position, int):
        input_grads.append((grad_position, inputs[grad_position]) if grad.return_ids else inputs[grad_position])

    param_grads = []
    if isinstance(weights, Parameter):
        param_grads.append(Tensor(dtype=weights.dtype, shape=weights.shape, init=Zero()))
    elif isinstance(weights, (ParameterTuple, list)):
        param_grads = [Tensor(dtype=weight.dtype, shape=weight.shape, init=Zero()) for weight in weights]
    grads_output = None
    get_all = input_grads and param_grads
    if get_all:
        grads_output = (_to_result(input_grads), _to_result(param_grads))
    elif input_grads:
        grads_output = _to_result(input_grads)
    elif param_grads:
        grads_output = _to_result(param_grads)

    if grad.get_value:
        return outputs, grads_output

    if grad.has_aux:
        aux_output = tuple(list(inputs)[1:])
        return grads_output, aux_output
    return grads_output
