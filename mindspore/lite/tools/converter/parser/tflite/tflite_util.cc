/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/tflite/tflite_util.h"
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
std::map<tflite::BuiltinOperator, std::string> tfMsOpTypeMap{
  {tflite::BuiltinOperator_CONV_2D, "Conv2D"},
  {tflite::BuiltinOperator_DEPTHWISE_CONV_2D, "DepthwiseConv2D"},
  {tflite::BuiltinOperator_AVERAGE_POOL_2D, "MeanPooling"},
  {tflite::BuiltinOperator_MAX_POOL_2D, "MaxPooling"},
  {tflite::BuiltinOperator_ADD, "Add"},
  {tflite::BuiltinOperator_CONCATENATION, "Concat"},
  {tflite::BuiltinOperator_RESIZE_BILINEAR, "ResizeBilinear"},
  {tflite::BuiltinOperator_RESHAPE, "Reshape"},
  {tflite::BuiltinOperator_LOGISTIC, "Logistic"},
  {tflite::BuiltinOperator_MUL, "Mul"},
  {tflite::BuiltinOperator_SOFTMAX, "Softmax"},
  {tflite::BuiltinOperator_FULLY_CONNECTED, "FullyConnected"},
  {tflite::BuiltinOperator_SLICE, "Slice"},
  {tflite::BuiltinOperator_SUB, "Sub"},
  {tflite::BuiltinOperator_TRANSPOSE, "Transpose"},
  {tflite::BuiltinOperator_PACK, "Stack"},
  {tflite::BuiltinOperator_MEAN, "Mean"},
  {tflite::BuiltinOperator_RELU6, "ReLU6"},
  {tflite::BuiltinOperator_TANH, "Tanh"},
  {tflite::BuiltinOperator_RSQRT, "Rsqrt"},
  {tflite::BuiltinOperator_ARG_MAX, "Argmax"},
  {tflite::BuiltinOperator_SQUARED_DIFFERENCE, "SquaredDifference"},
  {tflite::BuiltinOperator_FAKE_QUANT, "FakeQuant"},
  {tflite::BuiltinOperator_TRANSPOSE_CONV, "DeConv2D"},
  {tflite::BuiltinOperator_PAD, "Pad"},
  {tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, "NearestNeighbor"},
  {tflite::BuiltinOperator_RELU, "ReLU"},
  {tflite::BuiltinOperator_LEAKY_RELU, "LeakyRelu"},
  {tflite::BuiltinOperator_SQUEEZE, "Squeeze"},
  {tflite::BuiltinOperator_POW, "Pow"},
  {tflite::BuiltinOperator_ARG_MIN, "Argmin"},
  {tflite::BuiltinOperator_CEIL, "Ceil"},
  {tflite::BuiltinOperator_EXPAND_DIMS, "ExpandDims"},
  {tflite::BuiltinOperator_FILL, "Fill"},
  {tflite::BuiltinOperator_DIV, "Div"},
  {tflite::BuiltinOperator_FLOOR, "flOOR"},
  {tflite::BuiltinOperator_FLOOR_DIV, "FloorDiv"},
  {tflite::BuiltinOperator_FLOOR_MOD, "FloorMod"},
  {tflite::BuiltinOperator_GATHER, "Gather"},
  {tflite::BuiltinOperator_GATHER_ND, "GatherND"},
  {tflite::BuiltinOperator_REVERSE_V2, "reverse"},
  {tflite::BuiltinOperator_RANGE, "Range"},
  {tflite::BuiltinOperator_RANK, "Rank"},
  {tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION, "LocalResponseNorm"},
  {tflite::BuiltinOperator_GATHER, "Gather"},
  {tflite::BuiltinOperator_EXP, "Exp"},
  {tflite::BuiltinOperator_SPLIT_V, "SplitV"},
  {tflite::BuiltinOperator_SPLIT, "Split"},
  {tflite::BuiltinOperator_BATCH_TO_SPACE_ND, "BatchToSpaceND"},
  {tflite::BuiltinOperator_STRIDED_SLICE, "StridedSlice"},
  {tflite::BuiltinOperator_ONE_HOT, "OneHot"},
  {tflite::BuiltinOperator_SHAPE, "Shape"},
  {tflite::BuiltinOperator_SQUEEZE, "Squeeze"},
  {tflite::BuiltinOperator_ABS, "Abs"},
  {tflite::BuiltinOperator_SIN, "Sin"},
  {tflite::BuiltinOperator_COS, "Cos"},
  {tflite::BuiltinOperator_LOG, "Log"},
  {tflite::BuiltinOperator_SQRT, "Sqrt"},
  {tflite::BuiltinOperator_SQUARE, "Square"},
  {tflite::BuiltinOperator_LOGICAL_NOT, "LogicalNot"},
  {tflite::BuiltinOperator_LOGICAL_AND, "LogicalAnd"},
  {tflite::BuiltinOperator_LOGICAL_OR, "LogicalOr"},
  {tflite::BuiltinOperator_HARD_SWISH, "HSwish"},
  {tflite::BuiltinOperator_SUM, "Sum"},
  {tflite::BuiltinOperator_REDUCE_PROD, "ReduceProd"},
  {tflite::BuiltinOperator_REDUCE_MAX, "ReduceMax"},
  {tflite::BuiltinOperator_REDUCE_MIN, "ReduceMin"},
  // {tflite::BuiltinOperator_REDUCE_ANY, "ReduceAny"},
  {tflite::BuiltinOperator_SCATTER_ND, "ScatterNd"},
  {tflite::BuiltinOperator_MAXIMUM, "Maximum"},
  {tflite::BuiltinOperator_MINIMUM, "Minimum"},
  {tflite::BuiltinOperator_ADD_N, "AddN"},
  {tflite::BuiltinOperator_CAST, "Cast"},
  {tflite::BuiltinOperator_EQUAL, "Equal"},
  {tflite::BuiltinOperator_NOT_EQUAL, "NotEqual"},
  {tflite::BuiltinOperator_GREATER, "Greater"},
  {tflite::BuiltinOperator_GREATER_EQUAL, "GreaterEqual"},
  {tflite::BuiltinOperator_LESS, "Less"},
  {tflite::BuiltinOperator_LESS_EQUAL, "LessEqual"},
  {tflite::BuiltinOperator_DEPTH_TO_SPACE, "DepthToSpace"},
  {tflite::BuiltinOperator_SPACE_TO_BATCH_ND, "SpaceToBatchND"},
  {tflite::BuiltinOperator_SPACE_TO_DEPTH, "SpaceToDepth"},
  {tflite::BuiltinOperator_ROUND, "Round"},
  {tflite::BuiltinOperator_WHERE, "Where"},
  {tflite::BuiltinOperator_SPARSE_TO_DENSE, "SparseToDense"},
  {tflite::BuiltinOperator_ZEROS_LIKE, "ZerosLike"},
  {tflite::BuiltinOperator_TILE, "Tile"},
  {tflite::BuiltinOperator_TOPK_V2, "TopKV2"},
  {tflite::BuiltinOperator_REVERSE_SEQUENCE, "ReverseSequence"},
  {tflite::BuiltinOperator_UNIQUE, "Unique"},
  {tflite::BuiltinOperator_UNPACK, "Unstack"},
  {tflite::BuiltinOperator_CUSTOM, "Custom"},
  {tflite::BuiltinOperator_MIRROR_PAD, "MirrorPad"},
  {tflite::BuiltinOperator_NEG, "Neg"},
  {tflite::BuiltinOperator_PRELU, "PRELU"},
  {tflite::BuiltinOperator_HASHTABLE_LOOKUP, "HashtableLookup"},
  {tflite::BuiltinOperator_LSH_PROJECTION, "LshProjection"},
  {tflite::BuiltinOperator_SKIP_GRAM, "SKipGram"},
  {tflite::BuiltinOperator_WHILE, "While"},
};

std::map<tflite::ActivationFunctionType, mindspore::ActivationType> tfMsActivationFunctionMap{
  {tflite::ActivationFunctionType_NONE, mindspore::ActivationType::NO_ACTIVATION},
  {tflite::ActivationFunctionType_RELU, mindspore::ActivationType::RELU},
  {tflite::ActivationFunctionType_RELU6, mindspore::ActivationType::RELU6},
  {tflite::ActivationFunctionType_TANH, mindspore::ActivationType::TANH},
};

std::map<int, TypeId> type_map = {
  {tflite::TensorType_FLOAT64, TypeId::kNumberTypeFloat64},    {tflite::TensorType_FLOAT32, TypeId::kNumberTypeFloat32},
  {tflite::TensorType_FLOAT16, TypeId::kNumberTypeFloat16},    {tflite::TensorType_INT32, TypeId::kNumberTypeInt32},
  {tflite::TensorType_INT16, TypeId::kNumberTypeInt16},        {tflite::TensorType_INT8, TypeId::kNumberTypeInt8},
  {tflite::TensorType_INT64, TypeId::kNumberTypeInt64},        {tflite::TensorType_UINT8, TypeId::kNumberTypeUInt8},
  {tflite::TensorType_BOOL, TypeId::kNumberTypeBool},          {tflite::TensorType_STRING, TypeId::kObjectTypeString},
  {tflite::TensorType_COMPLEX64, TypeId::kNumberTypeComplex64}};

mindspore::ActivationType GetActivationFunctionType(tflite::ActivationFunctionType tfliteAFType) {
  return tfMsActivationFunctionMap.at(tfliteAFType);
}

std::string GetMSOpType(tflite::BuiltinOperator tfliteOpType) {
  auto iter = tfMsOpTypeMap.find(tfliteOpType);
  if (iter == tfMsOpTypeMap.end()) {
    return tflite::EnumNameBuiltinOperator(tfliteOpType);
  }
  return iter->second;
}

TypeId GetTfliteDataType(const tflite::TensorType &tflite_data_type) {
  auto iter = type_map.find(tflite_data_type);
  if (iter == type_map.end()) {
    return kTypeUnknown;
  }
  return iter->second;
}

std::string GetPadModeStr(tflite::Padding tflite_padmode) {
  if (tflite_padmode == tflite::Padding_SAME) {
    return "same";
  } else if (tflite_padmode == tflite::Padding_VALID) {
    return "valid";
  } else {
    return "pad";
  }
}

mindspore::PadMode GetPadMode(tflite::Padding tflite_padmode) {
  if (tflite_padmode == tflite::Padding_SAME) {
    return mindspore::PadMode::SAME;
  } else if (tflite_padmode == tflite::Padding_VALID) {
    return mindspore::PadMode::VALID;
  } else {
    return mindspore::PadMode::PAD;
  }
}

size_t GetDataTypeSize(const TypeId &data_type) {
  switch (data_type) {
    case TypeId::kNumberTypeFloat32:
      return sizeof(float);
    case TypeId::kNumberTypeFloat16:
      return sizeof(float) / 2;
    case TypeId::kNumberTypeInt8:
      return sizeof(int8_t);
    case TypeId::kNumberTypeInt32:
      return sizeof(int);
    case TypeId::kNumberTypeUInt8:
      return sizeof(uint8_t);
    case TypeId::kNumberTypeUInt32:
      return sizeof(uint32_t);
    case TypeId::kNumberTypeInt64:
      return sizeof(int64_t);
    default:
      MS_LOG(ERROR) << data_type << " is Unsupported datatype";
      return TypeId::kTypeUnknown;
  }
}

STATUS getPaddingParam(const std::unique_ptr<tflite::TensorT> &tensor, mindspore::PadMode pad_mode, int strideH,
                       int strideW, int windowH, int windowW, std::vector<int64_t> *params) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "the input tensor is null";
    return RET_ERROR;
  }
  if (tensor->shape.empty()) {
    MS_LOG(DEBUG) << "the tensor's shape is dynamic, which obtain only when running.";
    return RET_NO_CHANGE;
  }
  int padUp = 0;
  int padDown = 0;
  int padLeft = 0;
  int padRight = 0;
  if (pad_mode == mindspore::PadMode::SAME) {
    auto shape = tensor->shape;
    int H_input = shape.at(1);
    int W_input = shape.at(2);
    if (strideH == 0) {
      MS_LOG(ERROR) << "strideH is zero";
      return RET_ERROR;
    }
    int H_output = ceil(H_input * 1.0 / strideH);
    int pad_needed_H = (H_output - 1) * strideH + windowH - H_input;
    padUp = floor(pad_needed_H / 2.0);
    padDown = pad_needed_H - padUp;
    if (strideW == 0) {
      MS_LOG(ERROR) << "strideW is zero";
      return RET_ERROR;
    }
    int W_output = ceil(W_input * 1.0 / strideW);
    int pad_needed_W = (W_output - 1) * strideW + windowW - W_input;
    padLeft = floor(pad_needed_W / 2.0);
    padRight = pad_needed_W - padLeft;
  }

  params->emplace_back(padUp);
  params->emplace_back(padDown);
  params->emplace_back(padLeft);
  params->emplace_back(padRight);
  return RET_OK;
}

void Split(const std::string &src_str, std::vector<std::string> *dst_str, const std::string &chr) {
  MS_ASSERT(dst_str != nullptr);
  if (src_str.empty()) {
    MS_LOG(ERROR) << "src_str is empty";
    return;
  }
  std::string ::size_type p1 = 0, p2 = src_str.find(chr);
  while (std::string::npos != p2) {
    dst_str->push_back(src_str.substr(p1, p2 - p1));
    p1 = p2 + chr.size();
    p2 = src_str.find(chr, p1);
  }
  if (p1 != src_str.length()) {
    dst_str->push_back(src_str.substr(p1));
  }
}
}  // namespace lite
}  // namespace mindspore
