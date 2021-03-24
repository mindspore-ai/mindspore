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
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/base/tensorlist_getitem.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListGetItem;

namespace mindspore::kernel {

int TensorListGetItemCPUKernel::Init() { return RET_OK; }

int TensorListGetItemCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() >= 2);
  MS_ASSERT(in_tensors_.at(0) != nullptr);
  MS_ASSERT(in_tensors_.at(1) != nullptr);
  MS_ASSERT(out_tensors_.at(0) != nullptr);
  auto input0 = reinterpret_cast<lite::TensorList *>(in_tensors_.at(0));
  if (input0->root_tensor() != nullptr) {
    input0 = reinterpret_cast<lite::TensorList *>(input0->root_tensor());
  }
  dtype_ = input0->tensors_data_type();
  MS_ASSERT(in_tensors_.at(1)->data_c() != nullptr);
  index_ = reinterpret_cast<int *>(in_tensors_.at(1)->data_c())[0];
  int dim0 = input0->ElementsNum() - 1;
  if (index_ < 0 || index_ > dim0) {
    MS_LOG(ERROR) << "index tensor:[" << index_ << "] must be in [0, " << dim0 << "]!";
    return RET_ERROR;
  }
  auto src_ptr = input0->GetTensor(index_);
  MS_ASSERT(src_ptr != nullptr);
  if (src_ptr->data_type() != kTypeUnknown) {
    if (src_ptr->ElementsNum() != out_tensors_.at(0)->ElementsNum()) {
      MS_LOG(ERROR) << "src_ptr->ElementsNum():" << src_ptr->ElementsNum()
                    << " must be equal to out_tensors_[0]->ElementsNum():" << out_tensors_.at(0)->ElementsNum();
      return RET_ERROR;
    }
    auto status = lite::Tensor::CopyTensorData(*src_ptr, out_tensors_.at(0));
    if (status == RET_ERROR) {
      MS_LOG(ERROR) << "copy tensor data failed!";
      return RET_ERROR;
    }
  } else {
    // reset data buffer is zero
    auto out_data = out_tensors_[0]->data_c();
    if (out_data == nullptr) {
      MS_LOG(ERROR) << "data of out_tensors_[0] is nullptr";
      return RET_ERROR;
    }
    memset(out_data, 0, out_tensors_[0]->Size());
  }
  return RET_OK;
}

int TensorListGetItemCPUKernel::ReSize() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListGetItem, LiteKernelCreator<TensorListGetItemCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_TensorListGetItem, LiteKernelCreator<TensorListGetItemCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorListGetItem, LiteKernelCreator<TensorListGetItemCPUKernel>)
}  // namespace mindspore::kernel
