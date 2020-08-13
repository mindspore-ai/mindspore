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
#include "securec/include/securec.h"
#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_util.h"

namespace mindspore {
namespace lite {
STATUS TfliteNodeParser::CopyTfliteTensorData(const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                              const tflite::TensorT *tflite_tensor,
                                              schema::TensorT *tensor) {
  auto count = 1;
  std::for_each(tflite_tensor->shape.begin(), tflite_tensor->shape.end(), [&](int32_t sha) { count *= sha; });
  auto data_size = count * GetDataTypeSize(TypeId(tensor->dataType));
  auto buffer_idx = tflite_tensor->buffer;
  if (!tfliteModelBuffer[buffer_idx]->data.empty()) {
    tensor->data.resize(data_size);
    if (memcpy_s(tensor->data.data(), data_size, tfliteModelBuffer[buffer_idx]->data.data(), data_size)) {
      MS_LOG(ERROR) << "memcpy tensor data failed";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "src tensor data is empty";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TfliteNodeParser::ParseTensor(const std::vector<tflite::TensorT *> &ts,
                                     const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                     mindspore::lite::TensorCache *tensor_cache,
                                     int node_type,
                                     bool isWeight) {
  for (const auto &t : ts) {
    auto idx = tensor_cache->FindTensor(t->name);
    if (idx < 0) {
      std::unique_ptr<schema::TensorT> tensor(new schema::TensorT);
      tensor->dataType = GetTfliteDataType(t->type);
      tensor->dims = t->shape;

      if (isWeight) {
        tensor->format = schema::Format_KHWC;
      } else {
        tensor->format = schema::Format_NHWC;
      }

      if (t->buffer > 0) {
        CopyTfliteTensorData(tfliteModelBuffer, t, tensor.get());
      }

      MS_LOG(DEBUG) << "add tensor name: " << t->name.c_str();
      tensor_cache->AddTensor(t->name, tensor.release(), node_type);
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
