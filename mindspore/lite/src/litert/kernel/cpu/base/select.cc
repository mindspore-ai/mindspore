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
#include "src/litert/kernel/cpu/base/select.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/tensorlist.h"
#include "src/common/tensor_util.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Select;

namespace mindspore::kernel {
constexpr static int kConditionIdx = 0;
constexpr static int kFirstIdx = 1;
constexpr static int kSecondIdx = 2;

int SelectCPUKernel::Prepare() { return RET_OK; }

int SelectCPUKernel::ReSize() { return RET_OK; }

int CopyTensorData(lite::Tensor *dst_tensor, const lite::Tensor *src_tensor) {
  if (dst_tensor->data_type() != src_tensor->data_type() || dst_tensor->format() != src_tensor->format() ||
      !(dst_tensor->shape() == src_tensor->shape() || (dst_tensor->shape().empty() && src_tensor->shape().empty()))) {
    MS_LOG(ERROR) << "input tensor and output tensor is incompatible.";
    MS_LOG(ERROR) << "input tensor data_type: " << src_tensor->data_type() << " vs "
                  << "output tensor data_type: " << dst_tensor->data_type()
                  << "input tensor format: " << src_tensor->format() << " vs "
                  << "output tensor format: " << dst_tensor->format() << " input tensor shape: " << src_tensor->shape()
                  << " vs "
                  << "output tensor shape: " << dst_tensor->shape();
    return RET_ERROR;
  }
  if (src_tensor->allocator() == nullptr) {
    MS_LOG(ERROR) << "src_tensor allocator is nullptr.";
    return RET_ERROR;
  }

  CHECK_NULL_RETURN(src_tensor->data());
  CHECK_NULL_RETURN(dst_tensor->data());
  // need replace with increase data ref count
  MS_CHECK_FALSE(src_tensor->Size() == 0, RET_ERROR);
  (void)memcpy(dst_tensor->data(), src_tensor->data(), src_tensor->Size());
  return RET_OK;
}

int MoveTensorListData(lite::TensorList *dst_tensorlist, lite::TensorList *src_tensorlist) {
  int ret = lite::CopyTensorListTensorDataType(dst_tensorlist, src_tensorlist);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CopyTensorListTensorDataType failed.";
    return ret;
  }
  return lite::MoveTensorListTensorData(dst_tensorlist, src_tensorlist);
}

int MoveData(const std::vector<lite::Tensor *>::iterator &dst_begin,
             const std::vector<lite::Tensor *>::iterator &dst_end,
             const std::vector<lite::Tensor *>::iterator &src_begin,
             const std::vector<lite::Tensor *>::iterator &src_limit) {
  for (auto dst_iter = dst_begin, src_iter = src_begin; dst_iter != dst_end; dst_iter++, src_iter++) {
    if (src_iter == src_limit) {
      MS_LOG(ERROR) << "out of range of input tensor";
      return RET_ERROR;
    }
    auto *dst_tensor = *dst_iter;
    auto *src_tensor = *src_iter;
    if (dst_tensor == nullptr || src_tensor == nullptr) {
      MS_LOG(ERROR) << "input tensor or output tensor of merge is nullptr";
      return RET_ERROR;
    }
    lite::STATUS ret = RET_OK;
    if (src_tensor->IsConst() || src_tensor->IsGraphInput()) {
      MS_LOG(DEBUG) << "Carry const data and graph inputs.";
      dst_tensor->FreeData();
      dst_tensor->set_data(src_tensor->data());
      dst_tensor->set_own_data(false);
    } else {
      if (src_tensor->data_type() == kObjectTypeTensorType && dst_tensor->data_type() == kObjectTypeTensorType) {
        MS_LOG(DEBUG) << "Carry MoveTensorListData";
        ret = MoveTensorListData(reinterpret_cast<lite::TensorList *>(dst_tensor),
                                 reinterpret_cast<lite::TensorList *>(src_tensor));
      } else {
        MS_LOG(DEBUG) << "Carry CopyTensorData";
        ret = CopyTensorData(dst_tensor, src_tensor);
      }
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Move data failed : " << ret;
      return ret;
    }
  }
  return RET_OK;
}

template <typename T>
int SelectRun(std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors) {
  MS_CHECK_TRUE_MSG(in_tensors.at(kFirstIdx)->Size() == out_tensors.at(0)->Size(), RET_ERROR,
                    "The tensor size should be the same.");
  auto size = in_tensors.at(kFirstIdx)->ElementsNum();
  MS_CHECK_GT(size, 0, RET_ERROR);
  auto condition = static_cast<bool *>(in_tensors.at(kConditionIdx)->data());
  auto input1 = static_cast<T *>(in_tensors.at(kFirstIdx)->data());
  auto input2 = static_cast<T *>(in_tensors.at(kSecondIdx)->data());
  auto output = static_cast<T *>(out_tensors.at(0)->data());
  if (condition == nullptr || input1 == nullptr || input2 == nullptr || output == nullptr) {
    return RET_NULL_PTR;
  }
  for (int i = 0; i < size; i++) {
    output[i] = condition[i] ? input1[i] : input2[i];
  }
  return RET_OK;
}

// inputs: bool*1 true-data*n false-data*n
// output: data*n
int SelectCPUKernel::Run() {
  CHECK_LESS_RETURN(in_tensors_.size(), FOURTH_INPUT);
  MS_CHECK_TRUE_MSG(in_tensors_.size() == out_tensors_.size() * C2NUM + 1, RET_ERROR,
                    "The tensor number is incorrect.");
  auto bool_tensor = in_tensors_.front();
  CHECK_NULL_RETURN(bool_tensor);
  MS_CHECK_TRUE_MSG(bool_tensor->data_type() == kNumberTypeBool, RET_ERROR, "The tensor data type is invalid.");
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
    MS_CHECK_TRUE_MSG(bool_tensor->shape().size() == in_tensors_.at(kFirstIdx)->shape().size(), RET_ERROR,
                      "The tensor size should be the same.");
    for (size_t i = 0; i < in_tensors_.at(kFirstIdx)->shape().size(); i++) {
      if (bool_tensor->shape()[i] != in_tensors_.at(kFirstIdx)->shape()[i]) {
        MS_LOG(ERROR) << "Tensor shapes differ in dim: " << i << " in_tensors_.at(0): " << bool_tensor->shape()[i]
                      << " in_tensors_.at(1): " << in_tensors_.at(kFirstIdx)->shape()[i];
        return RET_ERROR;
      }
    }
    switch (in_tensors_.at(kFirstIdx)->data_type()) {
      case kNumberTypeInt32:
        if (SelectRun<int>(in_tensors_, out_tensors_) != RET_OK) {
          MS_LOG(ERROR) << "Select Run with integer failed";
          return RET_ERROR;
        }
        break;
      case kNumberTypeFloat32:
        if (SelectRun<float>(in_tensors_, out_tensors_) != RET_OK) {
          MS_LOG(ERROR) << "Select Run with float failed";
          return RET_ERROR;
        }
        break;
      default:
        MS_LOG(ERROR) << "Unsupported data type for select " << in_tensors_.at(kFirstIdx)->data_type();
        return RET_ERROR;
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Select, LiteKernelCreator<SelectCPUKernel>)
}  // namespace mindspore::kernel
