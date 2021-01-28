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

#include "src/runtime/kernel/arm/fp32/concat_fp32.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {
int ConcatCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConcatCPUKernel::ReSize() {
  concat_param_->axis_ =
    concat_param_->axis_ >= 0 ? concat_param_->axis_ : in_tensors_.front()->shape().size() + concat_param_->axis_;
  return RET_OK;
}

int ConcatCPUKernel::DoConcat(int task_id) {
  auto input_num = in_tensors_.size();
  std::vector<void *> inputs_addr(input_num, nullptr);
  std::vector<int *> inputs_output_shape(input_num + 1, nullptr);

  std::vector<std::vector<int>> shapes;
  for (size_t i = 0; i < input_num; ++i) {
    inputs_addr[i] = in_tensors_[i]->data_c();
    shapes.push_back(in_tensors_[i]->shape());
    inputs_output_shape[i] = shapes[i].data();
  }
  auto output_shape = out_tensors_.at(0)->shape();
  inputs_output_shape[input_num] = output_shape.data();
  auto output_addr = out_tensors_.at(0)->data_c();

  Concat(inputs_addr.data(), input_num, concat_param_->axis_, inputs_output_shape.data(), output_shape.size(),
         output_addr, task_id, op_parameter_->thread_num_, sizeof(float));
  return RET_OK;
}

int ConcatRun(void *cdata, int task_id) {
  auto concat_kernel = reinterpret_cast<ConcatCPUKernel *>(cdata);
  auto error_code = concat_kernel->DoConcat(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConcatRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConcatCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ConcatRun, this, op_parameter_->thread_num_);
  return error_code;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Concat, LiteKernelCreator<ConcatCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Concat, LiteKernelCreator<ConcatCPUKernel>)
}  // namespace mindspore::kernel
