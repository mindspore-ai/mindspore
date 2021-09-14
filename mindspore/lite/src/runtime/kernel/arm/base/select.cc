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
#include "src/runtime/kernel/arm/base/select.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#endif

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Select;

namespace mindspore::kernel {
constexpr static int kFirstIdx = 1;
constexpr static int kSecondIdx = 2;

int SelectCPUKernel::Init() { return RET_OK; }

int SelectCPUKernel::ReSize() { return RET_OK; }

// inputs: bool*1 true-data*n false-data*n
// output: data*n
int SelectCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() >= 3);
  MS_ASSERT(in_tensors_.size() == out_tensors_.size() * 2 + 1);
  auto bool_tensor = in_tensors_.front();
  MS_ASSERT(bool_tensor != nullptr);
  MS_ASSERT(bool_tensor->data_type() == kNumberTypeBool);
  if (bool_tensor->Size() == 1) {
    auto condition = static_cast<bool *>(bool_tensor->data());
    if (condition == nullptr) {
      MS_LOG(ERROR) << "data of bool tensor is nullptr";
      return lite::RET_NULL_PTR;
    }
    if (*condition) {
      auto ret = MoveData(this->out_tensors_.begin(), this->out_tensors_.end(), this->in_tensors_.begin() + 1,
                          this->in_tensors_.begin() + 1 + this->out_tensors_.size());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "carry data error : " << ret;
        return ret;
      }
    } else {
      auto ret = MoveData(this->out_tensors_.begin(), this->out_tensors_.end(),
                          this->in_tensors_.begin() + 1 + this->out_tensors_.size(),
                          this->in_tensors_.begin() + 1 + 2 * this->out_tensors_.size());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "carry data error : " << ret;
        return ret;
      }
    }
  } else {
    MS_ASSERT(bool_tensor->shape().size() == in_tensors_.at(1)->shape().size());
    for (size_t i = 0; i < in_tensors_.at(1)->shape().size(); i++) {
      if (bool_tensor->shape()[i] != in_tensors_.at(1)->shape()[i]) {
        MS_LOG(ERROR) << "Tensor shapes differ in dim: " << i << " in_tensors_.at(0): " << bool_tensor->shape()[i]
                      << " in_tensors_.at(1): " << in_tensors_.at(1)->shape()[i];
        return RET_ERROR;
      }
    }
    MS_ASSERT(in_tensors_.at(1)->Size() == out_tensors_.at(0)->Size());
    auto size = in_tensors_.at(1)->ElementsNum();
    auto condition = static_cast<bool *>(bool_tensor->data());
    auto input1 = static_cast<float *>(in_tensors_.at(kFirstIdx)->data());
    auto input2 = static_cast<float *>(in_tensors_.at(kSecondIdx)->data());
    auto output = static_cast<float *>(out_tensors_.at(0)->data());
    if (condition == nullptr || input1 == nullptr || input2 == nullptr || output == nullptr) {
      return RET_NULL_PTR;
    }
    for (int i = 0; i < size; i++) {
      output[i] = condition[i] ? input1[i] : input2[i];
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Select, LiteKernelCreator<SelectCPUKernel>)
}  // namespace mindspore::kernel
