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

#include "mindspore/lite/tools/converter/parser/tflite/tflite_util.h"
#include <map>
#include <string>
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {
std::map<tflite::ActivationFunctionType, schema::ActivationType> tfMsActivationFunctionMap{
  {tflite::ActivationFunctionType_NONE, schema::ActivationType_NO_ACTIVATION},
  {tflite::ActivationFunctionType_RELU, schema::ActivationType_RELU},
  {tflite::ActivationFunctionType_RELU6, schema::ActivationType_RELU6},
};

schema::ActivationType GetActivationFunctionType(tflite::ActivationFunctionType tfliteAFType) {
  return tfMsActivationFunctionMap.at(tfliteAFType);
}

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
  {tflite::BuiltinOperator_RELU6, "Relu6"},
  {tflite::BuiltinOperator_TANH, "Tanh"},
  {tflite::BuiltinOperator_RSQRT, "Rsqrt"},
  {tflite::BuiltinOperator_ARG_MAX, "Argmax"},
  {tflite::BuiltinOperator_SQUARED_DIFFERENCE, "SquaredDifference"},
  {tflite::BuiltinOperator_FAKE_QUANT, "FakeQuant"},
};

std::string GetMSOpType(tflite::BuiltinOperator tfliteOpType) {
  auto iter = tfMsOpTypeMap.find(tfliteOpType);
  if (iter == tfMsOpTypeMap.end()) {
    return "unsupported_op_type";
  }
  return iter->second;
}

std::map<int, TypeId> type_map = {
  {tflite::TensorType_FLOAT32, TypeId::kNumberTypeFloat32}, {tflite::TensorType_FLOAT16, TypeId::kNumberTypeFloat16},
  {tflite::TensorType_INT32, TypeId::kNumberTypeInt32},     {tflite::TensorType_UINT8, TypeId::kNumberTypeUInt8},
  {tflite::TensorType_INT16, TypeId::kNumberTypeInt16},
};

TypeId GetTfliteDataType(const tflite::TensorType &tflite_data_type) {
  auto iter = type_map.find(tflite_data_type);
  if (iter == type_map.end()) {
    return kTypeUnknown;
  }
  return iter->second;
}

schema::PadMode GetPadMode(tflite::Padding tflite_padmode) {
  if (tflite_padmode == tflite::Padding_SAME) {
    return schema::PadMode_SAME;
  } else if (tflite_padmode == tflite::Padding_VALID) {
    return schema::PadMode_VALID;
  } else {
    return schema::PadMode_NOTSET;
  }
}

size_t GetDataTypeSize(const TypeId &data_type) {
  switch (data_type) {
    case TypeId::kNumberTypeFloat32:
      return sizeof(float);
    case TypeId::kNumberTypeFloat16:
      return sizeof(float) >> 1;
    case TypeId::kNumberTypeInt8:
      return sizeof(int8_t);
    case TypeId::kNumberTypeInt32:
      return sizeof(int);
    case TypeId::kNumberTypeUInt8:
      return sizeof(uint8_t);
    case TypeId::kNumberTypeUInt32:
      return sizeof(uint32_t);
    default:
      MS_LOG(ERROR) << "unsupport datatype";
  }
}
}  // namespace lite
}  // namespace mindspore
