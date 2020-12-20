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

int MergeCPUKernel::FreeInWorkTensor() const {
  for (auto &in_tensor : this->in_tensors_) {
    MS_ASSERT(in_tensor != nullptr);
    if (in_tensor->IsConst() || in_tensor->IsGraphInput()) {
      continue;
    }
    if (in_tensor->ref_count() > 0) {
      in_tensor->set_ref_count(in_tensor->ref_count() - 1);
      if (in_tensor->ref_count() <= 0) {
        auto ret = in_tensor->FreeData();
        if (0 != ret) {
          MS_LOG(ERROR) << "Free tensor data failed";
          return ret;
        }
      }
    }
  }
  return RET_OK;
}

// if one of input of merge is const-tensor, merge is always ready, this will cause error.
bool MergeCPUKernel::IsReady(const std::vector<lite::Tensor *> &scope_tensors) {
  MS_ASSERT(in_tensors().size() == 2 * out_tensors().size());
  return std::all_of(this->in_tensors().begin(), this->in_tensors().begin() + in_tensors().size() / 2,
                     [&](lite::Tensor *kernel_in_tensor) {
                       return kernel_in_tensor->IsConst() || kernel_in_tensor->IsGraphInput() ||
                              kernel_in_tensor->ref_count() >= 1;
                     }) ||
         std::all_of(this->in_tensors().begin() + in_tensors().size() / 2, this->in_tensors().end(),
                     [&](lite::Tensor *kernel_in_tensor) {
                       return kernel_in_tensor->IsConst() || kernel_in_tensor->IsGraphInput() ||
                              kernel_in_tensor->ref_count() >= 1;
                     });
}

int MergeCPUKernel::Init() { return RET_OK; }

int MergeCPUKernel::ReSize() { return RET_ERROR; }

int MergeCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 2 * out_tensors_.size());
  int in_tesnor_part_one = 0;
  int in_tensor_part_two = out_tensors().size();
  if (in_tensors_[in_tesnor_part_one]->data_c() != nullptr) {
    for (size_t i = 0; i < out_tensors().size(); i++) {
      auto out_data = out_tensors_[i]->data_c();
      auto in_data = in_tensors_[i]->data_c();
      MS_ASSERT(in_data != nullptr);
      MS_ASSERT(out_data != nullptr);
      memcpy(out_data, in_data, in_tensors_[i]->Size());
    }
  }
  if (in_tensors_[in_tensor_part_two]->data_c() != nullptr) {
    for (size_t i = 0; i < out_tensors().size(); i++) {
      auto out_data = out_tensors_[i]->data_c();
      auto in_data = in_tensors_[i + in_tensor_part_two]->data_c();
      MS_ASSERT(in_data != nullptr);
      MS_ASSERT(out_data != nullptr);
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
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Merge, CpuMergeKernelCreator)
}  // namespace mindspore::kernel
