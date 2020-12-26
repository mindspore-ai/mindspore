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
#include "src/runtime/kernel/arm/fp32/slice_fp32.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/slice_fp32.h"
#include "src/ops/slice.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Slice;

namespace mindspore::kernel {
int SliceLaunch(void *cdata, int task_id) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "Input cdata is nullptr!";
    return RET_ERROR;
  }
  auto kernel = reinterpret_cast<SliceCPUKernel *>(cdata);
  return kernel->SliceParallelRun(task_id);
}

int SliceCPUKernel::ReSize() {
  auto primitive_slice = reinterpret_cast<const mindspore::lite::Slice *>(primitive_);
  auto begin = primitive_slice->GetPostProcessBegin();
  auto size = primitive_slice->GetPostProcessSize();

  param_->param_length_ = in_tensors_.at(0)->shape().size();
  if (param_->param_length_ > DIMENSION_4D) {
    MS_LOG(ERROR) << "input dimension num should <= " << DIMENSION_4D;
    return RET_ERROR;
  }
  for (int i = 0; i < param_->param_length_; ++i) {
    param_->shape_[i] = in_tensors_.at(0)->DimensionSize(i);
    param_->begin_[i] = begin.at(i);
    param_->size_[i] = size.at(i) < 0 ? param_->shape_[i] - param_->begin_[i] : size.at(i);
    param_->end_[i] = param_->begin_[i] + param_->size_[i];
  }
  if (param_->param_length_ < DIMENSION_4D) {
    PadSliceParameterTo4D(param_);
  }
  return RET_OK;
}

int SliceCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SliceCPUKernel::SliceParallelRun(int thread_id) {
  const float *input_data = reinterpret_cast<const float *>(in_tensors_.at(0)->MutableData());
  float *output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(input_data);
  MS_ASSERT(output_data);
  DoSlice(input_data, output_data, param_, thread_id);
  return RET_OK;
}

int SliceCPUKernel::Run() {
  auto ret = PreProcess();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreProcess fail!ret: " << ret;
    return ret;
  }
  const float *input_data = reinterpret_cast<const float *>(in_tensors_.at(0)->MutableData());
  float *output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  if (param_->size_[1] < op_parameter_->thread_num_) {
    DoSliceNoParallel(input_data, output_data, param_);
    return RET_OK;
  }
  ret = ParallelLaunch(this->context_->thread_pool_, SliceLaunch, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "slice launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Slice, LiteKernelCreator<SliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Slice, LiteKernelCreator<SliceCPUKernel>)
}  // namespace mindspore::kernel
