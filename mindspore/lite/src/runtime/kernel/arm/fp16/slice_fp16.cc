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

#include "src/runtime/kernel/arm/fp16/slice_fp16.h"
#include "src/kernel_registry.h"
#include "nnacl/base/slice_base.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore::kernel {
int SliceFp16Launch(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "Input cdata is nullptr!";
    return RET_ERROR;
  }
  auto kernel = reinterpret_cast<SliceFp16CPUKernel *>(cdata);
  return kernel->SliceFp16ParallelRun(task_id);
}

SliceFp16CPUKernel::~SliceFp16CPUKernel() {
  if (input_data_ != nullptr) {
    ms_context_->allocator->Free(input_data_);
    input_data_ = nullptr;
  }
}

int SliceFp16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  auto input_tensor = in_tensors_.at(0);
  if (input_tensor->data_type() == kNumberTypeFloat32 && input_tensor->data() != nullptr) {
    input_data_ =
      reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(input_tensor->ElementsNum() * sizeof(float16_t)));
    CHECK_NULL_RETURN(input_data_);
    Float32ToFloat16(reinterpret_cast<float *>(input_tensor->data()), input_data_, input_tensor->ElementsNum());
  }
  return SliceCPUKernel::Init();
}

int SliceFp16CPUKernel::SliceFp16ParallelRun(int thread_id) {
  void *input_data = input_data_ == nullptr ? in_tensors_.at(0)->data() : input_data_;
  CHECK_NULL_RETURN(input_data);
  DoSlice(input_data, out_tensors_.at(0)->data(), param_, thread_id, lite::DataTypeSize(kNumberTypeFloat16));
  return RET_OK;
}

int SliceFp16CPUKernel::Run() {
  void *input_data = input_data_ == nullptr ? in_tensors_.at(0)->data() : input_data_;
  CHECK_NULL_RETURN(input_data);
  CHECK_NULL_RETURN(out_tensors_.at(0)->data());
  if (param_->size_[1] < op_parameter_->thread_num_) {
    DoSliceNoParallel(input_data, out_tensors_.at(0)->data(), param_, lite::DataTypeSize(kNumberTypeFloat16));
    return RET_OK;
  }
  auto ret = ParallelLaunch(this->ms_context_, SliceFp16Launch, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "fp16 slice launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SliceFusion, LiteKernelCreator<SliceFp16CPUKernel>)
}  // namespace mindspore::kernel
