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
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SliceFusion;

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
  auto in_tensor = in_tensors_[0];
  auto begin_tensor = in_tensors_[1];
  auto size_tensor = in_tensors_[2];

  MS_ASSERT(in_tensor->shape().size() == begin_tensor->ElementsNum());
  MS_ASSERT(in_tensor->shape().size() == size_tensor->ElementsNum());
  MS_ASSERT(in_tensor->shape().size() <= DIMENSION_4D);

  auto begin = reinterpret_cast<int32_t *>(begin_tensor->data_c());
  auto size = reinterpret_cast<int32_t *>(size_tensor->data_c());

  param_->param_length_ = in_tensor->shape().size();
  if (param_->param_length_ > DIMENSION_4D) {
    MS_LOG(ERROR) << "input dimension num should <= " << DIMENSION_4D;
    return RET_ERROR;
  }
  for (int i = 0; i < param_->param_length_; ++i) {
    param_->shape_[i] = in_tensor->DimensionSize(i);
    param_->begin_[i] = begin[i];
    param_->size_[i] = size[i] < 0 ? param_->shape_[i] - param_->begin_[i] : size[i];
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
  DoSlice(in_tensors_.at(0)->data_c(), out_tensors_.at(0)->data_c(), param_, thread_id,
          lite::DataTypeSize(in_tensors_.at(0)->data_type()));
  return RET_OK;
}

int SliceCPUKernel::Run() {
  auto ret = PreProcess();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreProcess fail!ret: " << ret;
    return ret;
  }

  if (param_->size_[1] < op_parameter_->thread_num_) {
    DoSliceNoParallel(in_tensors_.at(0)->data_c(), out_tensors_.at(0)->data_c(), param_,
                      lite::DataTypeSize(in_tensors_.at(0)->data_type()));
    return RET_OK;
  }
  ret = ParallelLaunch(this->context_->thread_pool_, SliceLaunch, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "slice launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SliceFusion, LiteKernelCreator<SliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SliceFusion, LiteKernelCreator<SliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SliceFusion, LiteKernelCreator<SliceCPUKernel>)
}  // namespace mindspore::kernel
