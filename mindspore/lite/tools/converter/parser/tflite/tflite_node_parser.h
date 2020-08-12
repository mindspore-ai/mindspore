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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_NODE_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_NODE_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include "utils/log_adapter.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/parser/tflite/schema_generated.h"
#include "tools/common/tensor_util.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"
#include "tools/converter/parser/tflite/tflite_util.h"

namespace mindspore {
namespace lite {
class TfliteNodeParser {
 public:
  explicit TfliteNodeParser(const std::string &node_name) : name(node_name) {}

  virtual ~TfliteNodeParser() = default;

  virtual STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                       const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                       schema::CNodeT *op,
                       std::vector<int32_t> *tensors_id,
                       std::vector<schema::Format> *tensors_format,
                       std::map<int, int>  *tensors_id_map) = 0;

  void AddOpInput(schema::CNodeT *op,
                  std::vector<int32_t> *tensors_id,
                  std::vector<schema::Format> *tensors_format,
                  std::map<int, int> *tensors_id_map,
                  int idx, int new_idx, int total,  schema::Format format) {
    auto iter = tensors_id_map->find(idx);
    if (iter != tensors_id_map->end()) {
      op->inputIndex.emplace_back(iter->second);
    } else {
      if (idx < 0) {
        idx += total;
      }
      tensors_id->emplace_back(idx);
      tensors_format->emplace_back(format);
      tensors_id_map->insert(std::make_pair(idx, new_idx));
      op->inputIndex.emplace_back(new_idx);
    }
  }

  void AddOpOutput(schema::CNodeT *op,
                  std::vector<int32_t> *tensors_id,
                  std::vector<schema::Format> *tensors_format,
                  std::map<int, int> *tensors_id_map,
                  int idx, int new_idx, int total, schema::Format format) {
    auto iter = tensors_id_map->find(idx);
    if (iter != tensors_id_map->end()) {
      op->outputIndex.emplace_back(iter->second);
    } else {
      if (idx < 0) {
        idx += total;
      }
      tensors_id->emplace_back(idx);
      tensors_format->emplace_back(format);
      tensors_id_map->insert(std::make_pair(idx, new_idx));
      op->outputIndex.emplace_back(new_idx);
    }
  }

  template <typename T>
  STATUS GetTfliteData(const int32_t tensor_index,
                       const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                       std::vector<T> &attr_data) {
    int32_t count = 1;
    std::for_each(tflite_tensors[tensor_index]->shape.begin(), tflite_tensors[tensor_index]->shape.end(),
                  [&](int32_t sha) { count *= sha; });
    auto &buf_data = tflite_model_buffer[tflite_tensors[tensor_index]->buffer];
    if (buf_data == nullptr) {
      MS_LOG(ERROR) << "buf_data is null";
      return RET_NULL_PTR;
    }
    auto data_ptr = buf_data->data.data();
    switch (tflite_tensors[tensor_index]->type) {
      case tflite::TensorType_UINT8: {
        for (int i = 0; i < count; i++) {
          uint8_t data = *(static_cast<uint8_t *>(static_cast<void *>(data_ptr)));
          attr_data.emplace_back(static_cast<T>(data));
          data_ptr += sizeof(uint8_t);
        }
        break;
      }
      case tflite::TensorType_INT8: {
        for (int i = 0; i < count; i++) {
          int8_t data = *(static_cast<int8_t *>(static_cast<void *>(data_ptr)));
          attr_data.emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int8_t);
        }
        break;
      }
      case tflite::TensorType_INT16: {
        for (int i = 0; i < count; i++) {
          int16_t data = *(static_cast<int16_t *>(static_cast<void *>(data_ptr)));
          attr_data.emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int16_t);
        }
        break;
      }
      case tflite::TensorType_INT32: {
        for (int i = 0; i < count; i++) {
          int32_t data = *(static_cast<int32_t *>(static_cast<void *>(data_ptr)));
          attr_data.emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int32_t);
        }
        break;
      }
      case tflite::TensorType_INT64: {
        for (int i = 0; i < count; i++) {
          int64_t data = *(static_cast<int64_t *>(static_cast<void *>(data_ptr)));
          attr_data.emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int64_t);
        }
        break;
      }
      case tflite::TensorType_FLOAT32: {
        for (int i = 0; i < count; i++) {
          float data = *(static_cast<float *>(static_cast<void *>(data_ptr)));
          attr_data.emplace_back(static_cast<T>(data));
          data_ptr += sizeof(float);
        }
        break;
      }
      default: {
        MS_LOG(ERROR) << "wrong tensor type";
        return RET_ERROR;
      }
    }
    return RET_OK;
  }

 protected:
  const std::string &name;
  std::map<int, TypeId> dtype_map = {
    {tflite::TensorType_FLOAT64, TypeId::kNumberTypeFloat64},
    {tflite::TensorType_FLOAT32, TypeId::kNumberTypeFloat32},
    {tflite::TensorType_FLOAT16, TypeId::kNumberTypeFloat16},
    {tflite::TensorType_INT64, TypeId::kNumberTypeInt64},
    {tflite::TensorType_INT32, TypeId::kNumberTypeInt32},
    {tflite::TensorType_INT16, TypeId::kNumberTypeInt16},
    {tflite::TensorType_INT8, TypeId::kNumberTypeInt8},
    {tflite::TensorType_UINT8, TypeId::kNumberTypeUInt8},
    {tflite::TensorType_BOOL, TypeId::kNumberTypeBool},
  };
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_NODE_PARSER_H
