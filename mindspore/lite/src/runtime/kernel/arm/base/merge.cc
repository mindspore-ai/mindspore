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

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Merge;

namespace mindspore::kernel {
// if one of input of merge is const-tensor, merge is always ready, this will cause error.
bool MergeCPUKernel::IsReady() {
  MS_ASSERT(in_tensors().size() == 2);
  return std::any_of(this->in_tensors().begin(), this->in_tensors().end(), [&](lite::Tensor *kernel_in_tensor) {
    return kernel_in_tensor->IsConst() || kernel_in_tensor->ref_count() >= 1;
  });
}

int MergeCPUKernel::Init() { return RET_OK; }

int MergeCPUKernel::ReSize() { return RET_ERROR; }

int MergeCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 2);
  MS_ASSERT(out_tensors_.size() == 1);
  auto out_data = out_tensors_.front()->data_c();
  MS_ASSERT(out_data != nullptr);
  for (size_t i = 0; i < in_tensors().size(); i++) {
    if (in_tensors()[i]->data_c() != nullptr) {
      auto in_data = in_tensors_[i]->data_c();
      MS_ASSERT(in_data != nullptr);
      memcpy(out_data, in_data, in_tensors_[i]->Size());
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuMergeKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                          const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                          const lite::InnerContext *ctx, const KernelKey &desc,
                                          const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr";
    return nullptr;
  }
  if (desc.type != PrimitiveType_Merge) {
    MS_LOG(ERROR) << "type in desc is not Merge";
    free(parameter);
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "ctx is nullptr";
    free(parameter);
    return nullptr;
  }

  auto *kernel = new (std::nothrow) MergeCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
    free(parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Merge, CpuMergeKernelCreator)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Merge, CpuMergeKernelCreator)
}  // namespace mindspore::kernel
