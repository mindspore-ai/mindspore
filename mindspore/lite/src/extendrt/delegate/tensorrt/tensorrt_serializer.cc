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

#include "src/extendrt/delegate/tensorrt/tensorrt_serializer.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_runtime.h"
#include "src/common/file_utils.h"

namespace mindspore::lite {
nvinfer1::ICudaEngine *TensorRTSerializer::GetSerializedEngine() {
  if (serialize_file_path_.empty()) {
    return nullptr;
  }
  char *trt_model_stream{nullptr};
  size_t size{0};
  trt_model_stream = ReadFile(serialize_file_path_.c_str(), &size);
  if (trt_model_stream == nullptr || size == 0) {
    MS_LOG(WARNING) << "read engine file failed : " << serialize_file_path_;
    return nullptr;
  }
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger_);
  if (runtime == nullptr) {
    delete[] trt_model_stream;
    MS_LOG(ERROR) << "createInferRuntime failed.";
    return nullptr;
  }
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(trt_model_stream, size, nullptr);
  delete[] trt_model_stream;
  runtime->destroy();
  return engine;
}
void TensorRTSerializer::SaveSerializedEngine(nvinfer1::ICudaEngine *engine) {
  if (serialize_file_path_.size() == 0) {
    return;
  }
  nvinfer1::IHostMemory *ptr = engine->serialize();
  if (ptr == nullptr) {
    MS_LOG(ERROR) << "serialize engine failed";
    return;
  }

  int ret = WriteToBin(serialize_file_path_, ptr->data(), ptr->size());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "save engine failed " << serialize_file_path_;
  } else {
    MS_LOG(INFO) << "save engine to " << serialize_file_path_;
  }
  ptr->destroy();
  return;
}
}  // namespace mindspore::lite
