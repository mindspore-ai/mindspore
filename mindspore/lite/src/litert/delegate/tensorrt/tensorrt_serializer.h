/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_TENSORRT_SERIALIZER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_TENSORRT_SERIALIZER_H_
#include <string>
#include <utility>
#include <NvInfer.h>
#include "include/errorcode.h"
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"
#include "src/litert/delegate/tensorrt/tensorrt_runtime.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
class TensorRTSerializer {
 public:
  explicit TensorRTSerializer(const std::string &serialize_file_path)
      : serialize_file_path_(std::move(serialize_file_path)) {}

  ~TensorRTSerializer() = default;

  nvinfer1::ICudaEngine *GetSerializedEngine();

  void SaveSerializedEngine(nvinfer1::ICudaEngine *engine);

 private:
  std::string serialize_file_path_;
  TensorRTLogger logger_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_TENSORRT_SERIALIZER_H_
