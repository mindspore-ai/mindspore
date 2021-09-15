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

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {
int ConcatCPUKernel::Init() {
  MS_CHECK_TRUE_RET(in_tensors_.size() >= 1, RET_ERROR);
  CHECK_NULL_RETURN(in_tensors_.front());
  MS_CHECK_TRUE_RET(out_tensors_.size() == 1, RET_ERROR);
  CHECK_NULL_RETURN(out_tensors_.front());
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
    inputs_addr[i] = in_tensors_[i]->data();
    CHECK_NULL_RETURN(inputs_addr[i]);
    shapes.push_back(in_tensors_[i]->shape());
    MS_CHECK_LT(concat_param_->axis_, static_cast<int>(in_tensors_[i]->shape().size()), RET_ERROR);
    inputs_output_shape[i] = shapes[i].data();
  }
  auto output_shape = out_tensors_.at(0)->shape();
  MS_CHECK_LT(concat_param_->axis_, static_cast<int>(output_shape.size()), RET_ERROR);
  inputs_output_shape[input_num] = output_shape.data();
  auto output_addr = out_tensors_.at(0)->data();
  CHECK_NULL_RETURN(output_addr);

  MS_CHECK_FALSE_MSG(op_parameter_->thread_num_ == 0, RET_ERROR, "div zero");
  Concat(inputs_addr.data(), input_num, concat_param_->axis_, inputs_output_shape.data(), output_shape.size(),
         output_addr, task_id, op_parameter_->thread_num_, sizeof(float));
  return RET_OK;
}

int ConcatRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto concat_kernel = reinterpret_cast<ConcatCPUKernel *>(cdata);
  auto error_code = concat_kernel->DoConcat(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConcatRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConcatCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ConcatRun, this, op_parameter_->thread_num_);
  return error_code;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Concat, LiteKernelCreator<ConcatCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Concat, LiteKernelCreator<ConcatCPUKernel>)
}  // namespace mindspore::kernel
