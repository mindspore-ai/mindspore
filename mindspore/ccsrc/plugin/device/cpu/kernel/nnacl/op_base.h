/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_OP_BASE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_OP_BASE_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif

#define C0NUM 0
#define C1NUM 1
#define C2NUM 2
#define C3NUM 3
#define C4NUM 4
#define C5NUM 5
#define C6NUM 6
#define C7NUM 7
#define C8NUM 8
#define C9NUM 9
#define C10NUM 10
#define C11NUM 11
#define C12NUM 12
#define C13NUM 13
#define C14NUM 14
#define C16NUM 16
#define C20NUM 20
#define C21NUM 21
#define C24NUM 24
#define C28NUM 28
#define C32NUM 32
#define C36NUM 36
#define C40NUM 40
#define C44NUM 44
#define C48NUM 48
#define C56NUM 56
#define C64NUM 64
#define C128NUM 128
#define C150NUM 150
#define C256NUM 256
#define C512NUM 512
#define C1500NUM 1500
#define TILE_NUM 8

#define FP16_DATA_TYPE_LEN 2

#ifndef MS_UNLIKELY
#ifdef _MSC_VER
#define MS_UNLIKELY(x) (x)
#else
#define MS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#endif

#define MSMIN(x, y) ((x) < (y) ? (x) : (y))
#define MSMAX(x, y) ((x) > (y) ? (x) : (y))
#define MSCEIL(x) (int)((x) + (((x) - (int)(x)) > 0 ? 1 : 0))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))
#define DOWN_DIV(x, y) ((x) / (y))
#define DOWN_ROUND(x, y) ((x) / (y) * (y))

#define MSVALID(left, x, right) (MSMIN((MSMAX(left, x)), right))
#define SIZE_MUL_OVERFLOW(x, y) (((x) == 0) ? false : (SIZE_MAX / (x)) < (y))
#define INT_MUL_OVERFLOW(x, y)                                                                 \
  (((x) == 0) ? false                                                                          \
              : ((x) > 0 ? (((y) >= 0) ? (INT_MAX / (x)) < (y) : (INT_MAX / (x)) < (-1 * (y))) \
                         : (((y) >= 0) ? (INT_MAX / (x)) > (-1 * (y)) : (INT_MAX / (x)) > (y))))

#define INT_MUL_OVERFLOW_THRESHOLD(x, y, threshold)                                                    \
  (((x) == 0) ? false                                                                                  \
              : ((x) > 0 ? (((y) >= 0) ? ((threshold) / (x)) < (y) : ((threshold) / (x)) < (-1 * (y))) \
                         : (((y) >= 0) ? ((threshold) / (x)) > (-1 * (y)) : ((threshold) / (x)) > (y))))

#define INT_ADD_OVERFLOW(x, y) (INT_MAX - (x)) < (y)

#define INT_ADD_OVERFLOW_THRESHOLD(x, y, threshold) ((threshold) - (x)) < (y)

#define COMM_SHAPE_SIZE 4
#define MAX_SHAPE_SIZE 8

#define FIRST_INPUT 0
#define SECOND_INPUT 1
#define THIRD_INPUT 2
#define FOURTH_INPUT 3
#define FIFTH_INPUT 4
#define SIXTH_INPUT 5

#define DIMENSION_0D 0
#define DIMENSION_1D 1
#define DIMENSION_2D 2
#define DIMENSION_3D 3
#define DIMENSION_4D 4
#define DIMENSION_5D 5
#define DIMENSION_6D 6
#define DIMENSION_7D 7
#define DIMENSION_8D 8
#define DIMENSION_9D 9
#define DIMENSION_10D 10
#define DIMENSION_11D 11
#define kInputIndex 0
#define kWeightIndex 1
#define kBiasIndex 2
#define kOutputIndex 0
#define kNHWC_N 0
#define kNHWC_H 1
#define kNHWC_W 2
#define kNHWC_C 3
#define kNCHW_N 0
#define kNCHW_C 1
#define kNCHW_H 2
#define kNCHW_W 3
#define kHWCN_C 2
#define kHWNC_N 2
#define kHWCN_N 3
#define kNDHWC_N 0
#define kNDHWC_D 1
#define kNDHWC_H 2
#define kNDHWC_W 3
#define kNDHWC_C 4
#define kInputSize1 2
#define kInputSize2 3
#define MAX_AXIS_SIZE 6
#define MAX_LEN 256
#define FLT16_MAX 65504
#define kDefaulLiteMaxSpinCount 300000
#define kDefaulLiteMinSpinCount 1
#define kDefaulLiteIosSpinCount 1
#define DEFAULT_GROUP_NAME_LEN 101
#define kValueThreshold6 6

#define CLARGSINDEX0 0
#define CLARGSINDEX1 1
#define CLARGSINDEX2 2
#define CLARGSINDEX3 3
#define CLARGSINDEX4 4
#define CLARGSINDEX5 5
#define CLARGSINDEX6 6
#define CLARGSINDEX7 7
#define CLARGSINDEX8 8
#define CLARGSINDEX9 9

#define CLIDX_X 0
#define CLIDX_Y 1
#define CLIDX_Z 2
#define CLIDX_W 3

#define RELU6_MIN_VAL 0
#define RELU6_MAX_VAL 6

#if ENABLE_HIGH_PERFORMANCE
#define MS_CHECK_TRUE_RET(value, errcode)
#define MS_CHECK_TRUE_RET_VOID(value)
#define MS_CHECK_FALSE(value, errcode)
#define MS_CHECK_TRUE_MSG(value, errcode, msg)
#define MS_CHECK_FALSE_MSG(value, errcode, msg)
#define MS_CHECK_LT(value1, value2, errcode)
#define MS_CHECK_GT(value1, value2, errcode)
#define MS_CHECK_LE(value1, value2, errcode)
#define MS_CHECK_GE(value1, value2, errcode)
#define MS_CHECK_PTR_IF_NULL(ptr)

#define MS_CHECK_INT_MUL_NOT_OVERFLOW(value1, value2, errcode)
#define MS_CHECK_INT_ADD_NOT_OVERFLOW(value1, value2, errcode)

#define NNACL_CHECK_ZERO_RETURN_ERR(val)
#define NNACL_CHECK_ZERO_RETURN(val)
#define NNACL_CHECK_NULL_RETURN_ERR(ptr)
#define NNACL_CHECK_NULL_RETURN_VOID(ptr)
#define NNACL_CHECK_NULL_RETURN_NULL(ptr)
#else
// Check whether value is true, if not return 'errcode'
#define MS_CHECK_TRUE_RET(value, errcode) \
  do {                                    \
    if (!(value)) {                       \
      return errcode;                     \
    }                                     \
  } while (0)

#define MS_CHECK_TRUE_RET_VOID(value) \
  do {                                \
    if (!(value)) {                   \
      return;                         \
    }                                 \
  } while (0)

// Check whether value is false, if not return 'errcode'
#define MS_CHECK_FALSE(value, errcode) \
  do {                                 \
    if ((value)) {                     \
      return errcode;                  \
    }                                  \
  } while (0)

// Check whether value is true, if not return 'errcode'
// and print error string msg
#define MS_CHECK_TRUE_MSG(value, errcode, msg) \
  do {                                         \
    if (!(value)) {                            \
      MS_LOG(ERROR) << #msg;                   \
      return errcode;                          \
    }                                          \
  } while (0)

#define MS_CHECK_FALSE_MSG(value, errcode, msg) \
  do {                                          \
    if ((value)) {                              \
      MS_LOG(ERROR) << #msg;                    \
      return errcode;                           \
    }                                           \
  } while (0)

#define MS_CHECK_LT(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) >= (value2)) {                                                      \
      MS_LOG(ERROR) << "check ge fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_GT(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) <= (value2)) {                                                      \
      MS_LOG(ERROR) << "check gt fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_LE(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) > (value2)) {                                                       \
      MS_LOG(ERROR) << "check le fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_GE(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) < (value2)) {                                                       \
      MS_LOG(ERROR) << "check ge fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_EQ(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) != (value2)) {                                                      \
      MS_LOG(ERROR) << "check eq fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_PTR_IF_NULL(ptr)                                \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return;                                                    \
    }                                                            \
  } while (0)

#define MS_CHECK_INT_MUL_NOT_OVERFLOW(value1, value2, errcode) \
  MS_CHECK_TRUE_RET(!(INT_MUL_OVERFLOW(value1, value2)), errcode)
#define MS_CHECK_INT_ADD_NOT_OVERFLOW(value1, value2, errcode) \
  MS_CHECK_TRUE_RET(!(INT_ADD_OVERFLOW(value1, value2)), errcode)

#define NNACL_CHECK_ZERO_RETURN_ERR(val) \
  do {                                   \
    if ((val) == 0) {                    \
      return NNACL_ERR;                  \
    }                                    \
  } while (0)

#define NNACL_CHECK_ZERO_RETURN(val) \
  do {                               \
    if ((val) == 0) {                \
      return;                        \
    }                                \
  } while (0)

#define NNACL_CHECK_NULL_RETURN_ERR(ptr) \
  do {                                   \
    if ((ptr) == NULL) {                 \
      return NNACL_NULL_PTR;             \
    }                                    \
  } while (0)

#define NNACL_CHECK_NULL_RETURN_VOID(ptr) \
  do {                                    \
    if ((ptr) == NULL) {                  \
      return;                             \
    }                                     \
  } while (0)

#define NNACL_CHECK_NULL_RETURN_NULL(ptr) \
  do {                                    \
    if ((ptr) == NULL) {                  \
      return NULL;                        \
    }                                     \
  } while (0)
#endif

enum PrimType {
  PrimType_NONE = 0,
  PrimType_Abs = 1,
  PrimType_Activation = 2,
  PrimType_ActivationGrad = 3,
  PrimType_Adam = 4,
  PrimType_AddFusion = 5,
  PrimType_AdderFusion = 6,
  PrimType_AddGrad = 7,
  PrimType_AddN = 8,
  PrimType_All = 9,
  PrimType_ApplyMomentum = 10,
  PrimType_ArgMaxFusion = 11,
  PrimType_ArgMinFusion = 12,
  PrimType_Assert = 13,
  PrimType_Assign = 14,
  PrimType_AssignAdd = 15,
  PrimType_AudioSpectrogram = 16,
  PrimType_AvgPoolFusion = 17,
  PrimType_AvgPoolGrad = 18,
  PrimType_BatchNorm = 19,
  PrimType_BatchNormGrad = 20,
  PrimType_BatchToSpace = 21,
  PrimType_BatchToSpaceND = 22,
  PrimType_BiasAdd = 23,
  PrimType_BinaryCrossEntropy = 24,
  PrimType_BinaryCrossEntropyGrad = 25,
  PrimType_BiasAddGrad = 26,
  PrimType_BroadcastTo = 27,
  PrimType_Cast = 28,
  PrimType_Ceil = 29,
  PrimType_Clip = 30,
  PrimType_Concat = 31,
  PrimType_Attention = 32,
  PrimType_Conv2DBackpropFilterFusion = 33,
  PrimType_Conv2DBackpropInputFusion = 34,
  PrimType_Conv2DFusion = 35,
  PrimType_Conv2dTransposeFusion = 36,
  PrimType_Cos = 37,
  PrimType_ConstantOfShape = 38,
  PrimType_Crop = 39,
  PrimType_CustomExtractFeatures = 40,
  PrimType_CustomNormalize = 41,
  PrimType_CustomPredict = 42,
  PrimType_DeConv2DGradFilter = 43,
  PrimType_Depend = 44,
  PrimType_DepthToSpace = 45,
  PrimType_DetectionPostProcess = 46,
  PrimType_DivFusion = 47,
  PrimType_DivGrad = 48,
  PrimType_Dropout = 49,
  PrimType_DropoutGrad = 50,
  PrimType_Elu = 51,
  PrimType_Eltwise = 52,
  PrimType_Equal = 53,
  PrimType_EmbeddingLookupFusion = 54,
  PrimType_ExpFusion = 55,
  PrimType_ExpandDims = 56,
  PrimType_FakeQuantWithMinMaxVars = 57,
  PrimType_FakeQuantWithMinMaxVarsPerChannel = 58,
  PrimType_FftReal = 59,
  PrimType_FftImag = 60,
  PrimType_Flatten = 61,
  PrimType_FlattenGrad = 62,
  PrimType_Floor = 63,
  PrimType_FloorDiv = 64,
  PrimType_FloorMod = 65,
  PrimType_Fill = 66,
  PrimType_FullConnection = 67,
  PrimType_FusedBatchNorm = 68,
  PrimType_Gather = 69,
  PrimType_GatherNd = 70,
  PrimType_Greater = 71,
  PrimType_GreaterEqual = 72,
  PrimType_HashtableLookup = 73,
  PrimType_InstanceNorm = 74,
  PrimType_LayerNormFusion = 75,
  PrimType_LeakyRelu = 76,
  PrimType_Less = 77,
  PrimType_LessEqual = 78,
  PrimType_Log = 79,
  PrimType_LogGrad = 80,
  PrimType_LogicalAnd = 81,
  PrimType_LogicalNot = 82,
  PrimType_LogicalOr = 83,
  PrimType_LpNormalization = 84,
  PrimType_LRN = 85,
  PrimType_LshProjection = 86,
  PrimType_LSTM = 87,
  PrimType_L2NormalizeFusion = 88,
  PrimType_MatMulFusion = 89,
  PrimType_Maximum = 90,
  PrimType_MaximumGrad = 91,
  PrimType_MaxPoolFusion = 92,
  PrimType_MaxPoolGrad = 93,
  PrimType_Merge = 94,
  PrimType_Mfcc = 95,
  PrimType_Minimum = 96,
  PrimType_MinimumGrad = 97,
  PrimType_Mod = 98,
  PrimType_MulFusion = 99,
  PrimType_MulGrad = 100,
  PrimType_Neg = 101,
  PrimType_NegGrad = 102,
  PrimType_NotEqual = 103,
  PrimType_NonMaxSuppression = 104,
  PrimType_OneHot = 105,
  PrimType_OnesLike = 106,
  PrimType_PadFusion = 107,
  PrimType_PartialFusion = 108,
  PrimType_PowerGrad = 109,
  PrimType_PowFusion = 110,
  PrimType_PriorBox = 111,
  PrimType_PReLUFusion = 112,
  PrimType_QuantDTypeCast = 113,
  PrimType_Rank = 114,
  PrimType_Range = 115,
  PrimType_Reciprocal = 116,
  PrimType_RealDiv = 117,
  PrimType_ReduceFusion = 118,
  PrimType_Reshape = 119,
  PrimType_Resize = 120,
  PrimType_ReverseSequence = 121,
  PrimType_ReverseV2 = 122,
  PrimType_Rfft = 123,
  PrimType_ROIPooling = 124,
  PrimType_Round = 125,
  PrimType_Rsqrt = 126,
  PrimType_ScaleFusion = 127,
  PrimType_ScatterNd = 128,
  PrimType_SGD = 129,
  PrimType_Shape = 130,
  PrimType_SigmoidCrossEntropyWithLogits = 131,
  PrimType_SigmoidCrossEntropyWithLogitsGrad = 132,
  PrimType_Sin = 133,
  PrimType_SkipGram = 134,
  PrimType_SliceFusion = 135,
  PrimType_SmoothL1Loss = 136,
  PrimType_SmoothL1LossGrad = 137,
  PrimType_Softmax = 138,
  PrimType_SoftmaxCrossEntropyWithLogits = 139,
  PrimType_SpaceToBatch = 140,
  PrimType_SpaceToBatchND = 141,
  PrimType_SpaceToDepth = 142,
  PrimType_SparseSoftmaxCrossEntropyWithLogits = 143,
  PrimType_SparseToDense = 144,
  PrimType_Split = 145,
  PrimType_Sqrt = 146,
  PrimType_Squeeze = 147,
  PrimType_Square = 148,
  PrimType_SquaredDifference = 149,
  PrimType_Stack = 150,
  PrimType_StridedSlice = 151,
  PrimType_SubFusion = 152,
  PrimType_SubGrad = 153,
  PrimType_Switch = 154,
  PrimType_TensorListFromTensor = 155,
  PrimType_TensorListGetItem = 156,
  PrimType_TensorListReserve = 157,
  PrimType_TensorListSetItem = 158,
  PrimType_TensorListStack = 159,
  PrimType_TileFusion = 160,
  PrimType_TopKFusion = 161,
  PrimType_Transpose = 162,
  PrimType_Unique = 163,
  PrimType_UnsortedSegmentSum = 164,
  PrimType_Unsqueeze = 165,
  PrimType_Unstack = 166,
  PrimType_LSTMGrad = 167,
  PrimType_Where = 168,
  PrimType_ZerosLike = 169,
  PrimType_Select = 170,
  PrimType_ScatterNdUpdate = 171,
  PrimType_GRU = 172,
  PrimType_NonZero = 173,
  PrimType_InvertPermutation = 174,
  PrimType_Size = 175,
  PrimType_RandomStandardNormal = 176,
  PrimType_CropAndResize = 177,
  PrimType_Erf = 178,
  PrimType_StridedSliceGrad = 179,
  PrimType_IsFinite = 180,
  PrimType_LinSpace = 181,
  PrimType_UniformReal = 182,
  PrimType_AbsGrad = 183,
  PrimType_RsqrtGrad = 184,
  PrimType_SqrtGrad = 185,
  PrimType_LayerNormGrad = 186,
  PrimType_ResizeGrad = 187,
  PrimType_Splice = 188,
  PrimType_LogSoftmax = 189,
  PrimType_Call = 190,
  PrimType_Custom = 191,
  PrimType_CumSum = 192,
  PrimType_SplitWithOverlap = 193,
  PrimType_GenOP = 194,
  PrimType_RaggedRange = 195,
  PrimType_GLU = 196,
  PrimType_TensorArray = 197,
  PrimType_TensorArrayRead = 198,
  PrimType_TensorArrayWrite = 199,
  PrimType_Affine = 200,
  PrimType_AllGather = 201,
  PrimType_ReduceScatter = 202,
  PrimType_DynamicQuant = 203,
  PrimType_LSTMGradData = 204,
  PrimType_LSTMGradWeight = 205,
  PrimType_RandomNormal = 206,
  PrimType_NLLLoss = 207,
  PrimType_NLLLossGrad = 208,
  PrimType_FormatTranspose = 209,
  PrimType_GatherD = 210,
  PrimType_GroupNormFusion = 211,
  PrimType_Log1p = 212,
  PrimType_TensorScatterAdd = 213,
  PrimType_SparseFillEmptyRows = 214,
  PrimType_SparseReshape = 215,
  PrimType_SparseSegmentSum = 216,
  PrimType_ScatterElements = 217,
  PrimType_Triu = 218,
  PrimType_Tril = 219,
  PrimType_AdamWeightDecay = 220,
  PrimType_MIN = PrimType_NONE,
  PrimType_MAX = PrimType_AdamWeightDecay + 1,

  // inner operators.
  PrimType_Inner_ToFormat = 10000,
  PrimType_Inner_GltextureToOpencl = 10001,
  PrimType_Inner_Identity = 10002,
  PrimType_Inner_ShapeFusion = 10003,
  PrimType_Inner_GraphKernel = 10004,
  PrimType_Inner_SplitReduceConcatFusion = 10005,
  PrimType_Inner_EncoderLayer = 10006,
  PrimType_Inner_FseDecode = 10007,
  PrimType_InnerOpMax,
  PrimType_InnerOpMin = PrimType_Inner_ToFormat
};

typedef enum FormatC {
  DEFAULT_FORMAT = -1,
  Format_NCHW = 0,
  Format_NHWC = 1,
  Format_NHWC4 = 2,
  Format_HWKC = 3,
  Format_HWCK = 4,
  Format_KCHW = 5,
  Format_CKHW = 6,
  Format_KHWC = 7,
  Format_CHWK = 8,
  Format_HW = 9,
  Format_HW4 = 10,
  Format_NC = 11,
  Format_NC4 = 12,
  Format_NC4HW4 = 13,
  Format_NONE = 14,  // The origin Format_NUM_OF_FORMAT can't be used.
  Format_NCDHW = 15,
  Format_NWC = 16,
  Format_NCW = 17,
  Format_NDHWC = 18,
  Format_NC8HW8 = 19,
  Format_NC16HW16 = 20,
  Format_MAX,
  Format_MIN = Format_NCHW
} FormatC;

typedef enum TypeIdC {
  kTypeUnknown = 0,
  kMetaTypeBegin = kTypeUnknown,
  kMetaTypeType,  // Type
  kMetaTypeAny,
  kMetaTypeObject,
  kMetaTypeTypeType,  // TypeType
  kMetaTypeProblem,
  kMetaTypeExternal,
  kMetaTypeNone,
  kMetaTypeNull,
  kMetaTypeEllipsis,
  kMetaTypeEnd,
  //
  // Object types
  //
  kObjectTypeBegin = kMetaTypeEnd,
  kObjectTypeNumber,
  kObjectTypeString,
  kObjectTypeList,
  kObjectTypeTuple,
  kObjectTypeSlice,
  kObjectTypeKeyword,
  kObjectTypeTensorType,
  kObjectTypeRowTensorType,
  kObjectTypeCOOTensorType,
  kObjectTypeUndeterminedType,
  kObjectTypeClass,
  kObjectTypeDictionary,
  kObjectTypeFunction,
  kObjectTypeJTagged,
  kObjectTypeSymbolicKeyType,
  kObjectTypeEnvType,
  kObjectTypeRefKey,
  kObjectTypeRef,
  kObjectTypeEnd,
  //
  // Number Types
  //
  kNumberTypeBegin = kObjectTypeEnd,
  kNumberTypeBool,
  kNumberTypeInt,
  kNumberTypeInt8,
  kNumberTypeInt16,
  kNumberTypeInt32,
  kNumberTypeInt64,
  kNumberTypeUInt,
  kNumberTypeUInt8,
  kNumberTypeUInt16,
  kNumberTypeUInt32,
  kNumberTypeUInt64,
  kNumberTypeFloat,
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeComplex64,
  kNumberTypeComplex128,
  kNumberTypeEnd
} TypeIdC;

typedef enum DataOrder {
  RowMajor,
  ColMajor,
} DataOrder;

typedef struct OpParameter {
  char name_[100];
  int type_;
  int thread_num_;
  int quant_type_;
  bool is_train_session_;
  bool is_zero_shape_;
  void (*destroy_func_)(struct OpParameter *param);
} OpParameter;

typedef struct QuantArg {
  float scale_;
  int32_t zp_;
} QuantArg;

typedef struct QuantMulArg {
  int32_t multiplier_;
  int left_shift_;
  int right_shift_;
} QuantMulArg;

typedef enum ReductionType { Reduction_Sum, Reduction_Mean, Reduction_None } ReductionType;
typedef enum ActType {
  ActType_No = 0,
  ActType_Relu = 1,
  ActType_Sigmoid = 2,
  ActType_Relu6 = 3,
  ActType_Elu = 4,
  ActType_LeakyRelu = 5,
  ActType_Abs = 6,
  ActType_Relu1 = 7,
  ActType_Softsign = 8,
  ActType_Softplus = 9,
  ActType_Tanh = 10,
  ActType_Selu = 11,
  ActType_HSwish = 12,
  ActType_HSigmoid = 13,
  ActType_ThresholdRelu = 14,
  ActType_Linear = 15,
  ActType_HardTanh = 16,
  ActType_Sign = 17,
  ActType_Swish = 18,
  ActType_Gelu = 19,
  ActType_Unknown = 20
} ActType;
typedef enum PadMode { Pad_pad, Pad_same, Pad_valid } PadMode;
typedef enum RoundingMode { Rounding_No, Rounding_Away_from_zero, Rounding_Up } RoundingMode;
typedef enum CalFixedMultiplierMode {
  Method_No,
  Method_SinglePrecision,
  Method_DoublePrecision
} CalFixedMultiplierMode;

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_OP_BASE_H_
