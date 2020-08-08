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

#include <vector>
#include <memory>
#include <unordered_map>
#include "securec/include/securec.h"
#include "tools/converter/parser/tflite/tflite_node_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteNodeParser::CopyTfliteTensorData(const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                              const tflite::TensorT *tflite_tensor, schema::TensorT *tensor) {
  auto count = 1;
  std::for_each(tflite_tensor->shape.begin(), tflite_tensor->shape.end(), [&](int32_t sha) { count *= sha; });
  auto data_size = count * GetDataTypeSize(TypeId(tensor->dataType));
  auto buffer_idx = tflite_tensor->buffer;
  if (!tfliteModelBuffer[buffer_idx]->data.empty()) {
    tensor->data.resize(data_size);
    auto ret = memcpy_s(tensor->data.data(), data_size, tfliteModelBuffer[buffer_idx]->data.data(), data_size);
    if (ret) {
      MS_LOG(ERROR) << "memcpy tensor data failed, error code: %d" << ret;
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "src tensor data is empty.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TfliteNodeParser::ParseWeight(const std::vector<tflite::TensorT *> &weight_tenosrs,
                                     const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                     mindspore::lite::TensorCache *tensor_cache, schema::Format format) {
  for (const auto &weight_tensor : weight_tenosrs) {
    auto idx = tensor_cache->FindTensor(weight_tensor->name);
    if (idx < 0) {
      std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
      tensor->dataType = GetTfliteDataType(weight_tensor->type);
      tensor->dims = weight_tensor->shape;
      tensor->nodeType = schema::NodeType_ValueNode;
      // memcpy tensor data
      // buffer is 0 (which refers to an always existent empty buffer)
      if (weight_tensor->buffer > 0) {
        CopyTfliteTensorData(tfliteModelBuffer, weight_tensor, tensor.get());
      }
      MS_LOG(DEBUG) << "add weight tensor name: %s", weight_tensor->name.c_str();
      tensor_cache->AddTensor(weight_tensor->name, tensor.release(), TF_CONST);
    }
  }
  return RET_OK;
}

STATUS TfliteNodeParser::ParseBias(const std::vector<tflite::TensorT *> &bias_tensors,
                                   const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                   TensorCache *tensor_cache) {
  for (const auto &bias_tensor : bias_tensors) {
    auto idx = tensor_cache->FindTensor(bias_tensor->name);
    if (idx < 0) {
      std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
      tensor->dataType = GetTfliteDataType(bias_tensor->type);
      tensor->dims = bias_tensor->shape;
      tensor->nodeType = schema::NodeType_ValueNode;
      // memcpy tensor data
      // buffer is 0 (which refers to an always existent empty buffer)
      if (bias_tensor->buffer > 0) {
        CopyTfliteTensorData(tfliteModelBuffer, bias_tensor, tensor.get());
      }
      // MS_LOGD("add weight tensor name: %s", bias_tensor->name.c_str());
      tensor_cache->AddTensor(bias_tensor->name, tensor.release(), TF_CONST);
    }
  }
  return RET_OK;
}

STATUS TfliteNodeParser::ParseTensor(const std::vector<tflite::TensorT *> &ts,
                                     const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                     mindspore::lite::TensorCache *tensor_cache, int node_type,
                                     bool ifCopy) {
  for (const auto &t : ts) {
    auto idx = tensor_cache->FindTensor(t->name);
    if (idx < 0) {
      std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
      tensor->dataType = GetTfliteDataType(t->type);
      tensor->dims = t->shape;

      // memcpy tensor data, buffer is 0 (which refers to an always existent empty buffer)
      if (ifCopy && t->buffer > 0) {
        CopyTfliteTensorData(tfliteModelBuffer, t, tensor.get());
      }

      MS_LOG(DEBUG) << "add weight tensor name: %s", t->name.c_str();
      tensor_cache->AddTensor(t->name, tensor.release(), node_type);
    }
  }
  return RET_OK;
}

TypeId TfliteNodeParser::GetTfliteDataType(const tflite::TensorType &tflite_data_type) {
  static std::unordered_map<int, TypeId> type_map = {
    {tflite::TensorType_FLOAT32, TypeId::kNumberTypeFloat32}, {tflite::TensorType_FLOAT16, TypeId::kNumberTypeFloat16},
    {tflite::TensorType_INT32, TypeId::kNumberTypeInt32},     {tflite::TensorType_UINT8, TypeId::kNumberTypeUInt8},
    {tflite::TensorType_INT16, TypeId::kNumberTypeInt16},     {tflite::TensorType_INT8, TypeId::kNumberTypeInt8},
  };
  auto iter = type_map.find(tflite_data_type);
  if (iter == type_map.end()) {
    return kTypeUnknown;
  }
  return iter->second;
}
}  // namespace lite
}  // namespace mindspore
