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

#include "src/runtime/kernel/arm/base/switch.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/tensorlist.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Switch;

namespace mindspore::kernel {
int SwitchCPUKernel::PostProcess() {
  auto bool_tensor = in_tensors_.front();
  MS_ASSERT(bool_tensor != nullptr);
  MS_ASSERT(bool_tensor->data_type() == kNumberTypeBool);
  MS_ASSERT(bool_tensor->Size() == 1);
  MS_ASSERT(bool_tensor->Size() == 1);
  auto active = static_cast<bool *>(bool_tensor->data_c());
  if (active == nullptr) {
    MS_LOG(ERROR) << "data of bool tensor is nullptr";
    return lite::RET_NULL_PTR;
  }
  size_t in_index = 1;
  size_t out_index = (*active) ? 0 : (out_tensors_.size() / 2);
  while (in_index < in_tensors_.size()) {
    in_index++;
    auto out_tensor = out_tensors_.at(out_index++);
    out_tensor->ResetRefCount();
  }
  return FreeInWorkTensor();
}

int SwitchCPUKernel::Init() { return RET_OK; }

int SwitchCPUKernel::ReSize() { return RET_OK; }

// inputs: bool*1 data*n
// output: true-data*n, false-data*n
int SwitchCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() >= 2);
  auto bool_tensor = in_tensors_.front();
  MS_ASSERT(bool_tensor != nullptr);
  MS_ASSERT(bool_tensor->data_type() == kNumberTypeBool);
  MS_ASSERT(bool_tensor->Size() == 1);
  MS_ASSERT(bool_tensor->Size() == 1);
  auto active = static_cast<bool *>(bool_tensor->data_c());
  if (active == nullptr) {
    MS_LOG(ERROR) << "data of bool tensor is nullptr";
    return lite::RET_NULL_PTR;
  }
  size_t in_index = 1;
  size_t out_index = (*active) ? 0 : (out_tensors_.size() / 2);
  while (in_index < in_tensors_.size()) {
    auto in_tensor = in_tensors_.at(in_index++);
    auto out_tensor = out_tensors_.at(out_index++);
    // copy for tensorlist
    if (in_tensor->data_type() == kObjectTypeTensorType) {
      auto in_tensor_list = reinterpret_cast<lite::TensorList *>(in_tensor);
      auto out_tensor_list = reinterpret_cast<lite::TensorList *>(out_tensor);
      *out_tensor_list = *in_tensor_list;
      continue;
    }
    // copy for tensor
    MS_ASSERT(in_tensor != nullptr);
    MS_ASSERT(out_tensor != nullptr);
    auto input = in_tensor->data_c();
    auto output = out_tensor->data_c();
    MS_ASSERT(in_tensor->Size() == out_tensor->Size());
    if (input == nullptr || output == nullptr) {
      MS_LOG(ERROR) << "input tensor or output tensor have not been malloced";
      return lite::RET_NULL_PTR;
    }
    memcpy(output, input, in_tensor->Size());
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Switch, LiteKernelCreator<SwitchCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Switch, LiteKernelCreator<SwitchCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Switch, LiteKernelCreator<SwitchCPUKernel>)
}  // namespace mindspore::kernel
