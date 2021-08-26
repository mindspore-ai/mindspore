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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_BUILDER_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_BUILDER_
#include <NvInfer.h>
#include "include/errorcode.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "src/delegate/tensorrt/tensorrt_allocator.h"
#define MAX_BATCH_SIZE 64

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
class TensorRTLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
      MS_LOG(ERROR) << msg;
    } else if (severity == Severity::kWARNING) {
      MS_LOG(WARNING) << msg;
    } else if (severity == Severity::kINFO) {
      MS_LOG(INFO) << msg;
    } else {
      MS_LOG(DEBUG) << msg;
    }
  }
};

class TensorRTRuntime {
 public:
  TensorRTRuntime() = default;

  ~TensorRTRuntime();

  int Init();

  nvinfer1::IBuilder *GetBuilder() { return this->builder_; }

  int GetBatchSize() { return batch_size_; }

  void SetBatchSize(int batch_size) { batch_size_ = batch_size; }

  TensorRTAllocator *GetAllocator() { return this->allocator_; }

 private:
  bool is_init_ = false;
  nvinfer1::IBuilder *builder_{nullptr};
  TensorRTLogger logger_;
  TensorRTAllocator *allocator_{nullptr};
  int batch_size_{0};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_BUILDER_
