/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/kernel/cuda/cuda_kernel.h"

namespace mindspore::kernel {
int CudaKernel::PreProcess() {
  if (output_device_size_.size() == 0) {
    for (size_t i = 0; i < out_tensors_.size(); i++) {
      // allocator cudaMalloc mem_size: out_tensors_[i]->set_allocator(/*CudaAllocator*/)
      output_device_size_.push_back(helper_->GetOutputSizeList()[i]);
      output_device_ptrs_.push_back(out_tensors_[i]->MutableData());
    }
  } else {
    for (size_t i = 0; i < out_tensors_.size(); i++) {
      if (helper_->GetOutputSizeList()[i] > output_device_size_[i]) {
        out_tensors_[i]->FreeData();
        output_device_size_[i] = helper_->GetOutputSizeList()[i];
        output_device_ptrs_[i] = out_tensors_[i]->MutableData();
      }
    }
  }
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    input_device_ptrs_[i] = in_tensors_[i]->data();
  }
  return RET_OK;
}

int CudaKernel::ReSize() {
  // menory calculate
  std::vector<std::vector<size_t>> input_shapes;
  std::vector<std::vector<size_t>> output_shapes;
  for (auto in : in_tensors_) {
    std::vector<size_t> one_shape(in->shape().size());
    for (size_t i = 0; i < in->shape().size(); i++) {
      one_shape[i] = static_cast<size_t>(in->shape()[i]);
    }
    input_shapes.push_back(one_shape);
  }
  for (auto out : out_tensors_) {
    std::vector<size_t> one_shape(out->shape().size());
    for (size_t i = 0; i < out->shape().size(); i++) {
      one_shape[i] = static_cast<size_t>(out->shape()[i]);
    }
    output_shapes.push_back(one_shape);
  }
  helper_->ResetResource();
  auto ret = helper_->CalMemSize(input_shapes, output_shapes);
  CHECK_NOT_EQUAL_RETURN(ret, RET_OK);
  return RET_OK;
}
int CudaKernel::PostProcess() {
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    in_tensors_[i]->DecRefCount();
  }
  return RET_OK;
}
CudaKernel::~CudaKernel() { helper_ = nullptr; }
}  // namespace mindspore::kernel
