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

#include "src/runtime/kernel/arm/base/merge.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/tensorlist.h"
#include "src/common/utils.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Merge;

namespace mindspore::kernel {
int MergeCPUKernel::FreeInWorkTensor() const {
  size_t stride = in_tensors_.size() / 2;
  if (this->ready_part_ == LEFT_INPUT_PART) {
    for (size_t i = 0; i < stride; ++i) {
      auto in_tensor = in_tensors_[i];
      MS_ASSERT(in_tensor != nullptr);
      if (in_tensor->root_tensor() == in_tensor) {
        continue;
      }
      in_tensor->DecRefCount();
    }
  }
  if (this->ready_part_ == RIGHT_INPUT_PART) {
    for (size_t i = stride; i < in_tensors_.size(); ++i) {
      auto in_tensor = in_tensors_[i];
      MS_ASSERT(in_tensor != nullptr);
      if (in_tensor->root_tensor() == in_tensor) {
        continue;
      }
      in_tensor->DecRefCount();
    }
  }
  return RET_OK;
}

bool MergeCPUKernel::IsReady(const std::vector<lite::Tensor *> &scope_tensors) {
  auto ready_part = FindReadyPart(scope_tensors);
  return ready_part == LEFT_INPUT_PART || ready_part == RIGHT_INPUT_PART;
}

int MergeCPUKernel::Init() {
  MS_ASSERT(in_tensors_.size() == 2 * out_tensors_.size());
  size_t stride = in_tensors_.size() / 2;
  for (size_t i = 0; i < in_tensors_.size() / 2; i++) {
    MS_ASSERT(in_tensors_[i] != nullptr);
    MS_ASSERT(in_tensors_[i + stride] != nullptr);
    if (in_tensors_[i] == in_tensors_[i + stride]) {
      auto ret = in_tensors_[i]->set_root_tensor(in_tensors_[i]);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set root tensor for tensor(" << in_tensors_[i]->tensor_name() << ") failed";
        return ret;
      }
      ret = in_tensors_[i + stride]->set_root_tensor(in_tensors_[i + stride]);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set root tensor for tensor(" << in_tensors_[i + stride]->tensor_name() << ") failed";
        return ret;
      }
    }
  }
  return RET_OK;
}

int MergeCPUKernel::ReSize() { return RET_OK; }

InputPart MergeCPUKernel::FindReadyPart(const std::vector<lite::Tensor *> &scope_tensors) {
  MS_ASSERT(in_tensors_.size() == 2 * out_tensors_.size());
  bool is_root_tensor_ready =
    std::all_of(this->in_tensors().begin(), this->in_tensors().end(), [&](lite::Tensor *in_tensor) {
      // if not in scope_tensors, not care
      if (!IsContain(scope_tensors, in_tensor)) {
        return true;
      }
      // if not a root_tensor, not care
      if (in_tensor->root_tensor() == nullptr || in_tensor->root_tensor() != in_tensor) {
        return true;
      }
      return in_tensor->IsReady();
    });
  // check if all root tensor is ready
  if (!is_root_tensor_ready) {
    return UNKNOWN_INPUT_PART;
  }
  // check one part of in tensors of merge is ready
  // if not in scope_tensors, not care
  // if in scope_tensors, in_tensor need to be ready
  if (std::all_of(
        this->in_tensors().begin() + in_tensors().size() / 2, this->in_tensors().end(),
        [&](lite::Tensor *in_tensor) { return !IsContain(scope_tensors, in_tensor) || in_tensor->IsReady(); })) {
    return RIGHT_INPUT_PART;
  }
  if (std::all_of(
        this->in_tensors().begin(), this->in_tensors().begin() + in_tensors().size() / 2,
        [&](lite::Tensor *in_tensor) { return !IsContain(scope_tensors, in_tensor) || in_tensor->IsReady(); })) {
    return LEFT_INPUT_PART;
  }
  return UNKNOWN_INPUT_PART;
}

int MergeCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 2 * out_tensors_.size());
  ready_part_ = FindReadyPart(this->in_tensors_);
  if (ready_part_ == LEFT_INPUT_PART) {
    auto ret = MoveData(this->out_tensors_.begin(), this->out_tensors_.end(), this->in_tensors_.begin(),
                        this->in_tensors_.end());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "carry data error : " << ret;
      return ret;
    }
  } else if (ready_part_ == RIGHT_INPUT_PART) {
    auto ret = MoveData(this->out_tensors_.begin(), this->out_tensors_.end(),
                        (this->in_tensors_.begin() + in_tensors_.size() / 2), this->in_tensors_.end());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "carry data error : " << ret;
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "none input part of merge is ready";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Merge, LiteKernelCreator<MergeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Merge, LiteKernelCreator<MergeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Merge, LiteKernelCreator<MergeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Merge, LiteKernelCreator<MergeCPUKernel>)
}  // namespace mindspore::kernel
