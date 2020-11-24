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
#include "src/ops/primitive_c.h"
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"
#include "schema/schema_generated.h"
#include "tools/common/tensor_util.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"
#include "tools/converter/parser/tflite/tflite_util.h"

namespace mindspore::lite {
class TfliteNodeParser {
 public:
  explicit TfliteNodeParser(const std::string &node_name) : name(node_name) {}

  virtual ~TfliteNodeParser() = default;

  virtual lite::PrimitiveC *ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                               const std::unique_ptr<tflite::ModelT> &tflite_model) {
    return nullptr;
  }

  template <typename T>
  STATUS GetTfliteData(const int32_t tensor_index, const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                       std::vector<T> &attr_data) {
    const auto &tensor = tflite_tensors[tensor_index];
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "tensor is null";
      return RET_NULL_PTR;
    }

    int32_t count = 1;
    std::for_each(tensor->shape.begin(), tensor->shape.end(), [&](int32_t sha) { count *= sha; });
    auto &buf_data = tflite_model_buffer[tensor->buffer];
    if (buf_data == nullptr) {
      MS_LOG(ERROR) << "buf_data is null";
      return RET_NULL_PTR;
    }
    auto data_ptr = buf_data->data.data();
    if (data_ptr == nullptr) {
      MS_LOG(DEBUG) << "data is not a constant";
      return RET_NO_CHANGE;
    }
    switch (tensor->type) {
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
        MS_LOG(ERROR) << "wrong tensor type : " << tensor->type;
        return RET_ERROR;
      }
    }
    return RET_OK;
  }

 protected:
  const std::string &name;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_NODE_PARSER_H
