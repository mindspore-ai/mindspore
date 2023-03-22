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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_NODE_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_NODE_PARSER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"
#include "schema/schema_generated.h"
#include "tools/common/tensor_util.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"
#include "tools/converter/parser/tflite/tflite_util.h"
#include "ops/primitive_c.h"
#include "src/common/log_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
class TfliteNodeParser {
 public:
  explicit TfliteNodeParser(const std::string &node_name) : name(node_name) {}

  virtual ~TfliteNodeParser() = default;

  virtual PrimitiveCPtr Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                              const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                              const std::unique_ptr<tflite::ModelT> &tflite_model) {
    return nullptr;
  }

  template <typename T>
  STATUS GetTfliteData(const int32_t tensor_index, const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                       std::vector<T> *attr_data) {
    MS_ASSERT(attr_data != nullptr);
    CHECK_LESS_RETURN(tflite_tensors.size(), static_cast<size_t>(tensor_index + 1));
    const auto &tensor = tflite_tensors[tensor_index];
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "tensor is null";
      return RET_NULL_PTR;
    }

    size_t count = 1;
    std::for_each(tensor->shape.begin(), tensor->shape.end(), [&](int32_t sha) {
      MS_ASSERT(sha >= 0);
      count *= static_cast<size_t>(sha);
    });
    CHECK_LESS_RETURN(tflite_model_buffer.size(), static_cast<size_t>(tensor->buffer + 1));
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
        CHECK_LESS_RETURN(buf_data->data.size(), count * sizeof(uint8_t));
        for (size_t i = 0; i < count; i++) {
          auto data = *(static_cast<uint8_t *>(static_cast<void *>(data_ptr)));
          attr_data->emplace_back(static_cast<T>(data));
          data_ptr += sizeof(uint8_t);
        }
        break;
      }
      case tflite::TensorType_INT8: {
        CHECK_LESS_RETURN(buf_data->data.size(), count * sizeof(int8_t));
        for (size_t i = 0; i < count; i++) {
          auto data = *(static_cast<int8_t *>(static_cast<void *>(data_ptr)));
          attr_data->emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int8_t);
        }
        break;
      }
      case tflite::TensorType_INT16: {
        CHECK_LESS_RETURN(buf_data->data.size(), count * sizeof(int16_t));
        for (size_t i = 0; i < count; i++) {
          auto data = *(static_cast<int16_t *>(static_cast<void *>(data_ptr)));
          attr_data->emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int16_t);
        }
        break;
      }
      case tflite::TensorType_INT32: {
        CHECK_LESS_RETURN(buf_data->data.size(), count * sizeof(int32_t));
        for (size_t i = 0; i < count; i++) {
          auto data = *(static_cast<int32_t *>(static_cast<void *>(data_ptr)));
          attr_data->emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int32_t);
        }
        break;
      }
      case tflite::TensorType_INT64: {
        CHECK_LESS_RETURN(buf_data->data.size(), count * sizeof(int64_t));
        for (size_t i = 0; i < count; i++) {
          auto data = *(static_cast<int64_t *>(static_cast<void *>(data_ptr)));
          attr_data->emplace_back(static_cast<T>(data));
          data_ptr += sizeof(int64_t);
        }
        break;
      }
      case tflite::TensorType_FLOAT32: {
        CHECK_LESS_RETURN(buf_data->data.size(), count * sizeof(float));
        for (size_t i = 0; i < count; i++) {
          auto data = *(static_cast<float *>(static_cast<void *>(data_ptr)));
          attr_data->emplace_back(static_cast<T>(data));
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

  template <typename T>
  STATUS TransTfliteDataToVec2D(const int32_t tensor_index,
                                const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                std::vector<std::vector<T>> *vec) {
    MS_ASSERT(vec != nullptr);
    CHECK_LESS_RETURN(tflite_tensors.size(), static_cast<size_t>(tensor_index + 1));
    const auto &tensor = tflite_tensors[tensor_index];
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "tensor is null";
      return RET_NULL_PTR;
    }

    size_t count = 1;
    std::for_each(tensor->shape.begin(), tensor->shape.end(), [&](int32_t sha) {
      MS_ASSERT(sha >= 0);
      count *= static_cast<size_t>(sha);
    });
    CHECK_LESS_RETURN(tflite_model_buffer.size(), static_cast<size_t>(tensor->buffer + 1));
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

    constexpr size_t k2DMultipler = 2;
    (*vec).resize(count / 2, std::vector<T>(2));
    switch (tensor->type) {
      case tflite::TensorType_UINT8: {
        for (size_t i = 0; i < count / k2DMultipler; i++) {
          auto data = *(static_cast<uint8_t *>(static_cast<void *>(data_ptr + k2DMultipler * i * sizeof(uint8_t))));
          (*vec)[i][0] = static_cast<T>(data);
          data = *(static_cast<uint8_t *>(static_cast<void *>(data_ptr + (k2DMultipler * i + 1) * sizeof(uint8_t))));
          (*vec)[i][1] = static_cast<T>(data);
          i += 2;
        }
        break;
      }
      case tflite::TensorType_INT8: {
        for (size_t i = 0; i < count / k2DMultipler; i++) {
          auto data = *(static_cast<int8_t *>(static_cast<void *>(data_ptr + k2DMultipler * i * sizeof(int8_t))));
          (*vec)[i][0] = static_cast<T>(data);
          data = *(static_cast<int8_t *>(static_cast<void *>(data_ptr + (k2DMultipler * i + 1) * sizeof(int8_t))));
          (*vec)[i][1] = static_cast<T>(data);
        }
        break;
      }
      case tflite::TensorType_INT16: {
        for (size_t i = 0; i < count / k2DMultipler; i++) {
          auto data = *(static_cast<int16_t *>(static_cast<void *>(data_ptr + k2DMultipler * i * sizeof(int16_t))));
          (*vec)[i][0] = static_cast<T>(data);
          data = *(static_cast<int16_t *>(static_cast<void *>(data_ptr + (k2DMultipler * i + 1) * sizeof(int16_t))));
          (*vec)[i][1] = static_cast<T>(data);
        }
        break;
      }
      case tflite::TensorType_INT32: {
        for (size_t i = 0; i < count / k2DMultipler; i++) {
          auto data = *(static_cast<int32_t *>(static_cast<void *>(data_ptr + k2DMultipler * i * sizeof(int32_t))));
          (*vec)[i][0] = static_cast<T>(data);
          data = *(static_cast<int32_t *>(static_cast<void *>(data_ptr + (k2DMultipler * i + 1) * sizeof(int32_t))));
          (*vec)[i][1] = static_cast<T>(data);
        }
        break;
      }
      case tflite::TensorType_INT64: {
        for (size_t i = 0; i < count / k2DMultipler; i++) {
          auto data = *(static_cast<int64_t *>(static_cast<void *>(data_ptr + k2DMultipler * i * sizeof(int64_t))));
          (*vec)[i][0] = static_cast<T>(data);
          data = *(static_cast<int64_t *>(static_cast<void *>(data_ptr + (k2DMultipler * i + 1) * sizeof(int64_t))));
          (*vec)[i][1] = static_cast<T>(data);
        }
        break;
      }
      case tflite::TensorType_FLOAT32: {
        for (size_t i = 0; i < count / k2DMultipler; i++) {
          auto data = *(static_cast<float *>(static_cast<void *>(data_ptr + k2DMultipler * i * sizeof(float))));
          (*vec)[i][0] = static_cast<T>(data);
          data = *(static_cast<float *>(static_cast<void *>(data_ptr + (k2DMultipler * i + 1) * sizeof(float))));
          (*vec)[i][1] = static_cast<T>(data);
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
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_NODE_PARSER_H_
