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

#include "src/runtime/kernel/arm/int8/tanh_int8.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int TanhInt8CPUKernel::Init() {
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);

  tanh_quant_.in_scale_ = input->quant_params().front().scale;
  tanh_quant_.in_zp_ = input->quant_params().front().zeroPoint;
  tanh_quant_.out_scale_ = output->quant_params().front().scale;
  tanh_quant_.out_zp_ = output->quant_params().front().zeroPoint;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TanhInt8CPUKernel::ReSize() {
  element_size_ = in_tensors_.at(0)->ElementsNum();
  thread_count_ = MSMIN(element_size_, op_parameter_->thread_num_);
  thread_stride_ = UP_DIV(element_size_, thread_count_);
  return RET_OK;
}

int TanhInt8CPUKernel::DoActivation(int task_id) const {
  int current_size = element_size_ - task_id * thread_stride_;
  current_size = MSMIN(thread_stride_, current_size);
  if (current_size <= 0) {
    return RET_OK;
  }

  int8_t *cur_input = in_ptr_ + task_id * thread_stride_;
  int8_t *cur_output = out_ptr_ + task_id * thread_stride_;

  TanhInt8(cur_input, cur_output, current_size, &tanh_quant_);
  return RET_OK;
}

int TanhInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto activation_kernel = reinterpret_cast<TanhInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "TanhInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int TanhInt8CPUKernel::Run() {
  in_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  out_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());

  auto ret = ParallelLaunch(this->ms_context_, TanhInt8Run, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TanhInt8 Run failed";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
