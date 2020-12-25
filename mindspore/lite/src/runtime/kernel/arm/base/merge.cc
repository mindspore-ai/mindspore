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
                              kernel_in_tensor->ref_count() >= 1 ||
                              (kernel_in_tensor->data_type() == kObjectTypeTensorType);
                     });
}

int MergeCPUKernel::Init() { return RET_OK; }

int MergeCPUKernel::ReSize() { return RET_OK; }

bool MergeCPUKernel::PartialInputReady(int num_begin, int num_end) {
  MS_ASSERT(in_tensors_.size() == 2 * out_tensors_.size());
  bool result = (std::all_of(this->in_tensors().begin() + num_begin, this->in_tensors().begin() + num_end,
                             [&](lite::Tensor *kernel_in_tensor) {
                               return kernel_in_tensor->IsConst() || kernel_in_tensor->ref_count() >= 1 ||
                                      kernel_in_tensor->IsGraphInput() ||
                                      kernel_in_tensor->data_type() == kObjectTypeTensorType;
                             })) &&
                std::all_of(this->in_tensors_.begin() + num_begin, this->in_tensors_.begin() + num_end,
                            [&](lite::Tensor *in_tensor) {
                              if (in_tensor->data_type() != kObjectTypeTensorType) {
                                return in_tensor->data_c() != nullptr;
                              } else {
                                return true;
                              }
                            });
  return result;
}

int MergeCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 2 * out_tensors_.size());
  int in_tesnor_part_one = 0;
  int in_tensor_part_two = in_tensors_.size() / 2;
  int in_tensor_part_three = in_tensors_.size();
  if (PartialInputReady(in_tesnor_part_one, in_tensor_part_two)) {
    for (size_t i = 0; i < out_tensors().size(); i++) {
      auto out_data = out_tensors_[i]->data_c();
      auto in_data = in_tensors_[i]->data_c();
      if (in_tensors_[i]->data_type() == kObjectTypeTensorType) {
        auto in_tensor_list = reinterpret_cast<lite::TensorList *>(in_tensors_[i]);
        auto out_tensor_list = reinterpret_cast<lite::TensorList *>(out_tensors_[i]);
        if (std::any_of(in_tensor_list->tensors().begin(), in_tensor_list->tensors().end(),
                        [&](lite::Tensor *tensor) { return tensor->data_c() == nullptr; })) {
          continue;
        }
        *out_tensor_list = *in_tensor_list;
        continue;
      }
      MS_ASSERT(in_data != nullptr);
      MS_ASSERT(out_data != nullptr);
      memcpy(out_data, in_data, in_tensors_[i]->Size());
    }
  }
  if (PartialInputReady(in_tensor_part_two, in_tensor_part_three)) {
    for (size_t i = 0; i < out_tensors().size(); i++) {
      auto out_data = out_tensors_[i]->data_c();
      auto in_data = in_tensors_[i + in_tensor_part_two]->data_c();
      if (in_tensors_[i]->data_type() == kObjectTypeTensorType) {
        auto in_tensor_list = reinterpret_cast<lite::TensorList *>(in_tensors_[i + in_tensor_part_two]);
        auto out_tensor_list = reinterpret_cast<lite::TensorList *>(out_tensors_[i]);
        if (std::any_of(in_tensor_list->tensors().begin(), in_tensor_list->tensors().end(),
                        [&](lite::Tensor *tensor) { return tensor->data_c() == nullptr; })) {
          continue;
        }
        *out_tensor_list = *in_tensor_list;
        continue;
      }
      MS_ASSERT(in_data != nullptr);
      MS_ASSERT(out_data != nullptr);
      memcpy(out_data, in_data, in_tensors_[i]->Size());
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Merge, LiteKernelCreator<MergeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Merge, LiteKernelCreator<MergeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Merge, LiteKernelCreator<MergeCPUKernel>)
}  // namespace mindspore::kernel
