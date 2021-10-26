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
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
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
  MSLITE_CHECK_PTR(tensor);
  MSLITE_CHECK_PTR(params);
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
    MS_CHECK_TRUE_RET(shape.size() == DIMENSION_4D, RET_ERROR);
    int H_input = shape.at(1);
    int W_input = shape.at(2);
    if (strideH == 0) {
      MS_LOG(ERROR) << "strideH is zero";
      return RET_ERROR;
    }
    int H_output = ceil(H_input * 1.0 / strideH);
    if (INT_MUL_OVERFLOW(H_output - 1, strideH)) {
      MS_LOG(ERROR) << "data_size overflow";
      return RET_ERROR;
    }
    int pad_needed_H = (H_output - 1) * strideH + windowH - H_input;
    padUp = floor(pad_needed_H / 2.0);
    padDown = pad_needed_H - padUp;
    if (strideW == 0) {
      MS_LOG(ERROR) << "strideW is zero";
      return RET_ERROR;
    }
    int W_output = ceil(W_input * 1.0 / strideW);
    if (INT_MUL_OVERFLOW(W_output - 1, strideW)) {
      MS_LOG(ERROR) << "data_size overflow";
      return RET_ERROR;
    }
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
}  // namespace lite
}  // namespace mindspore
