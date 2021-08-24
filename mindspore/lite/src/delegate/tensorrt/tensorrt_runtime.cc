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

#include "src/delegate/tensorrt/tensorrt_runtime.h"
#include <mutex>
#include <vector>

namespace mindspore::lite {
int TensorRTRuntime::Init() {
  if (is_init_) {
    return RET_OK;
  }
  builder_ = nvinfer1::createInferBuilder(this->logger_);
  if (builder_ == nullptr) {
    MS_LOG(ERROR) << "create infer builder failed.";
    return RET_ERROR;
  }
  builder_->setMaxBatchSize(MAX_BATCH_SIZE);
  allocator_ = new (std::nothrow) TensorRTAllocator();
  if (allocator_ == nullptr) {
    MS_LOG(ERROR) << "Create allocator failed.";
    return RET_ERROR;
  }
  is_init_ = true;
  return RET_OK;
}

TensorRTRuntime::~TensorRTRuntime() {
  if (builder_ != nullptr) {
    builder_->destroy();
    builder_ = nullptr;
  }
  if (allocator_ != nullptr) {
    allocator_->ClearDeviceMem();
    delete allocator_;
    allocator_ = nullptr;
  }
}
}  // namespace mindspore::lite
