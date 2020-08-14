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
#include <memory>
#include "utils/log_adapter.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/parser/tflite/tflite_util.h"
#include "tools/converter/parser/tflite/schema_generated.h"
#include "tools/common/tensor_util.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
class TfliteNodeParser {
 public:
  explicit TfliteNodeParser(const std::string &nodeName) : name(nodeName) {}

  virtual ~TfliteNodeParser() = default;

  virtual STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                       schema::CNodeT *op,
                       TensorCache *tensor_cache,
                       bool quantizedModel) = 0;

  STATUS ParseTensor(const std::vector<tflite::TensorT *> &ts,
                     const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                     mindspore::lite::TensorCache *tensor_cache,
                     int node_type,
                     bool isWeight);

  STATUS CopyTfliteTensorData(const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                              const tflite::TensorT *tflite_tensor,
                              schema::TensorT *tensor);

  template <typename T>
  STATUS GetTfliteData(const int32_t tensor_index, const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                       std::vector<T> &attr_data) {
    int32_t count = 1;
    std::for_each(tfliteTensors[tensor_index]->shape.begin(), tfliteTensors[tensor_index]->shape.end(),
                  [&](int32_t sha) { count *= sha; });
    auto &buf_data = tfliteModelBuffer[tfliteTensors[tensor_index]->buffer];
    if (buf_data == nullptr) {
      MS_LOG(ERROR) << "buf_data is null";
      return RET_NULL_PTR;
    }
    auto data_ptr = buf_data->data.data();
    switch (tfliteTensors[tensor_index]->type) {
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
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_NODE_PARSER_H
