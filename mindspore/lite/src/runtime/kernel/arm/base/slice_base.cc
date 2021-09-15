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
#include "src/runtime/kernel/arm/base/slice_base.h"
#include "src/kernel_registry.h"
#include "nnacl/base/slice_base.h"
#include "src/tensor.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SliceFusion;

namespace {
constexpr int kNumInput0 = 0;
constexpr int kNumInput1 = 1;
constexpr int kNumInput2 = 2;
constexpr int kNumInputSize = 3;
}  // namespace
namespace mindspore::kernel {
int SliceLaunch(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "Input cdata is nullptr!";
    return RET_ERROR;
  }
  auto kernel = reinterpret_cast<SliceCPUKernel *>(cdata);
  return kernel->SliceParallelRun(task_id);
}

int SliceCPUKernel::ReSize() {
  auto in_tensor = in_tensors_[kNumInput0];
  auto begin_tensor = in_tensors_[kNumInput1];
  auto size_tensor = in_tensors_[kNumInput2];
  MS_ASSERT(in_tensor->shape().size() == static_cast<size_t>(begin_tensor->ElementsNum()));
  MS_ASSERT(in_tensor->shape().size() == static_cast<size_t>(size_tensor->ElementsNum()));
  MS_ASSERT(in_tensor->shape().size() <= DIMENSION_8D);
  auto begin = reinterpret_cast<int32_t *>(begin_tensor->data());
  CHECK_NULL_RETURN(begin);
  auto size = reinterpret_cast<int32_t *>(size_tensor->data());
  CHECK_NULL_RETURN(size);

  param_->param_length_ = in_tensor->shape().size();
  if (param_->param_length_ > DIMENSION_8D) {
    MS_LOG(ERROR) << "input dimension num should <= " << DIMENSION_8D;
    return RET_ERROR;
  }
  for (int i = 0; i < param_->param_length_; ++i) {
    param_->shape_[i] = in_tensor->DimensionSize(i);
    param_->begin_[i] = begin[i];
    param_->size_[i] = size[i] < 0 ? param_->shape_[i] - param_->begin_[i] : size[i];
    param_->end_[i] = param_->begin_[i] + param_->size_[i];
  }
  if (param_->param_length_ < DIMENSION_8D) {
    PadSliceParameterTo8D(param_);
  }
  return RET_OK;
}

int SliceCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), kNumInputSize);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[kNumInput0]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput1]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput2]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(op_parameter_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SliceCPUKernel::SliceParallelRun(int thread_id) {
  DoSlice(in_tensors_.at(0)->data(), out_tensors_.at(0)->data(), param_, thread_id,
          lite::DataTypeSize(in_tensors_.at(0)->data_type()));
  return RET_OK;
}

int SliceCPUKernel::Run() {
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->data());
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  if (input_data == nullptr || output_data == nullptr) {
    return RET_NULL_PTR;
  }
  // param_ shape info has already been extended to 8d
  constexpr size_t kDimHUnder8D = 5;
  if (param_->size_[kDimHUnder8D] < op_parameter_->thread_num_) {
    DoSliceNoParallel(input_data, output_data, param_, lite::DataTypeSize(in_tensors_.at(0)->data_type()));
    return RET_OK;
  }
  auto ret = ParallelLaunch(this->ms_context_, SliceLaunch, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "slice launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SliceFusion, LiteKernelCreator<SliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SliceFusion, LiteKernelCreator<SliceCPUKernel>)
}  // namespace mindspore::kernel
