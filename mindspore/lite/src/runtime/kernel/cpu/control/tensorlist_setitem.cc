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
 *
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "src/runtime/kernel/cpu/control/tensorlist_setitem.h"
#include "include/errorcode.h"
#include "src/runtime/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListSetItem;
namespace {
constexpr int kNumInputSize = 3;
constexpr int kNumInput2 = 2;
}  // namespace
namespace mindspore::kernel {
int TensorListSetItemCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kNumInputSize);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(in_tensors_.at(kNumInput2));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return RET_OK;
}

int TensorListSetItemCPUKernel::CheckParam() {
  if (in_tensors_[1]->data_type() != kNumberTypeInt && in_tensors_[1]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "in_tensors_[1]->data_type():" << in_tensors_[1]->data_type() << " must be int";
    return RET_ERROR;
  }
  if (in_tensors_[1]->ElementsNum() != 1) {
    MS_LOG(ERROR) << "in_tensors_[1]->ElementsNum():" << in_tensors_[1]->ElementsNum() << " must be equal to 1!";
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorListSetItemCPUKernel::IncrementOutputSize(int origin_size) {
  int new_tensors_size = origin_size + 1;
  output0_->set_shape({new_tensors_size});
  std::vector<std::vector<int>> out_shape;
  out_shape.resize(new_tensors_size, in_tensors_[2]->shape());
  auto ret = output0_->MallocTensorListData(in_tensors_[2]->data_type(), out_shape);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "increment output size malloc tensorlist data error";
    return ret;
  }
  return RET_OK;
}

int TensorListSetItemCPUKernel::Run() {
  input0_ = reinterpret_cast<lite::TensorList *>(in_tensors_[0]);
  CHECK_NULL_RETURN(input0_);
  output0_ = reinterpret_cast<lite::TensorList *>(out_tensors_[0]);
  CHECK_NULL_RETURN(output0_);
  if (CheckParam() != RET_OK) {
    MS_LOG(ERROR) << "check param failed.";
    return RET_ERROR;
  }

  int dim0 = output0_->ElementsNum() - 1;
  index_ = reinterpret_cast<int *>(in_tensors_[1]->data())[0];
  if (index_ < 0 || index_ > dim0) {
    if (IncrementOutputSize(output0_->tensors().size()) != RET_OK) {
      MS_LOG(ERROR) << "Resizeoutput Error ,index tensor:[" << index_ << "] must be in [0, " << dim0 << "]!";
      return RET_ERROR;
    }
  }
  input2_ = in_tensors_[2];
  if (!input0_->IsCompatibleShape(input2_->shape())) {
    return RET_ERROR;
  }
  output0_ = reinterpret_cast<lite::TensorList *>(out_tensors_[0]);
  MS_ASSERT(output0_ != nullptr);
  output0_->set_allocator(ms_context_->allocator);
  // new loop count
  if (output0_->tensors().empty() && input0_->tensors().empty()) {
    if (IncrementOutputSize(0) != RET_OK) {
      MS_LOG(ERROR) << "Resizeoutput Error!";
      return RET_ERROR;
    }
  }
  // copy each tensor in tensors_
  if (input0_->tensors().empty() && index_ == 0) {
    input0_->set_element_shape(input2_->shape());
    output0_->set_element_shape(input2_->shape());
  }
  if (output0_->allocator() == nullptr) {
    output0_->set_allocator(ms_context_->allocator);
  }
  for (int i = 0; i < output0_->ElementsNum(); ++i) {
    if (i == index_) {
      auto dst = output0_->GetTensor(i);
      if (dst == nullptr) {
        dst = lite::Tensor::CopyTensor(*input2_, true, ms_context_->allocator);
        auto tensors = output0_->tensors();
        tensors.emplace_back(dst);
        output0_->set_tensors(tensors);
      } else {
        dst->set_data_type(input2_->data_type());
        dst->set_shape(input2_->shape());
        dst->set_format(input2_->format());
        dst->set_category(input2_->category());
        dst->set_quant_clusters(input2_->quant_clusters());
        auto ret = lite::Tensor::CopyTensorData(*input2_, dst);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "CopyTensorData[" << i << "] is failed!";
          return RET_ERROR;
        }
      }
    } else {
      auto src = input0_->GetTensor(i);
      auto dst = output0_->GetTensor(i);
      if (src == nullptr) {
        MS_LOG(ERROR) << "src is nullptr.";
        return RET_NULL_PTR;
      }
      // merge move data will delete tensors
      if (dst == nullptr) {
        dst = lite::Tensor::CopyTensor(*src, src->data() != nullptr, ms_context_->allocator);
        auto tensors = output0_->tensors();
        tensors.emplace_back(dst);
        output0_->set_tensors(tensors);
        continue;
      }

      if (src->data_type() != kTypeUnknown) {
        auto ret = lite::Tensor::CopyTensorData(*src, dst);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "CopyTensorData[" << i << "] is failed!";
          return RET_ERROR;
        }
      }
    }
  }
  return RET_OK;
}

int TensorListSetItemCPUKernel::ReSize() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListSetItem, LiteKernelCreator<TensorListSetItemCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_TensorListSetItem, LiteKernelCreator<TensorListSetItemCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorListSetItem, LiteKernelCreator<TensorListSetItemCPUKernel>)
}  // namespace mindspore::kernel
