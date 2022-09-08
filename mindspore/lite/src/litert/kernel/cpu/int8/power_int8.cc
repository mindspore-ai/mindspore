/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/int8/power_int8.h"
#include <limits>
#include "nnacl/int8/power_int8.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowFusion;

namespace mindspore::kernel {
int PowerInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  MSLITE_CHECK_PTR(input);
  MSLITE_CHECK_PTR(output);

  auto in_quant_args = input->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  param_->quant_arg_.in_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  param_->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto out_quant_args = output->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  param_->quant_arg_.out_args_.scale_ = static_cast<float>(out_quant_args.front().scale);
  param_->quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  param_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PowerInt8CPUKernel::ReSize() { return RET_OK; }

int PowerInt8CPUKernel::DoPower(int task_id) {
  auto size = in_tensors_.at(0)->ElementsNum();
  MS_CHECK_GT(size, 0, RET_ERROR);
  int stride = UP_DIV(size, op_parameter_->thread_num_);
  int count = MSMIN(stride, size - stride * task_id);
  int8_t *cur_exp = nullptr;
  if (param_->broadcast_) {
    cur_exp = exp_ptr_;
  } else {
    cur_exp = exp_ptr_ + stride * task_id;
  }
  auto ret = PowerInt8(input_data_ + stride * task_id, cur_exp, output_data_ + stride * task_id, count, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerInt8 error ,task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int PowerInt8Run(void *cdata, int task_id, float, float) {
  auto power_kernel = reinterpret_cast<PowerInt8CPUKernel *>(cdata);
  auto ret = power_kernel->DoPower(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoPower error, task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int PowerInt8CPUKernel::Run() {
  MSLITE_CHECK_PTR(in_tensors_[0]);
  input_data_ = reinterpret_cast<int8_t *>(in_tensors_[0]->data());
  MSLITE_CHECK_PTR(input_data_);
  MSLITE_CHECK_PTR(out_tensors_[0]);
  output_data_ = reinterpret_cast<int8_t *>(out_tensors_[0]->data());
  MSLITE_CHECK_PTR(output_data_);
  auto exp_tensor = in_tensors_.at(1);
  MSLITE_CHECK_PTR(exp_tensor);
  auto exp_quant_args = exp_tensor->quant_params();
  if (exp_quant_args.size() < 1) {
    MS_LOG(ERROR) << "exp_tensor->quant_params().size() must be greater than 0";
    return RET_ERROR;
  }
  MSLITE_CHECK_PTR(param_);
  param_->quant_arg_.exp_args_.scale_ = static_cast<float>(exp_quant_args.front().scale);
  param_->quant_arg_.exp_args_.zp_ = exp_quant_args.front().zeroPoint;
  param_->broadcast_ = in_tensors_[0]->shape() == in_tensors_[1]->shape() ? false : true;
  exp_ptr_ = reinterpret_cast<int8_t *>(exp_tensor->MutableData());
  auto ret = ParallelLaunch(this->ms_context_, PowerInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerInt8Run error, error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_PowFusion, LiteKernelCreator<PowerInt8CPUKernel>)
}  // namespace mindspore::kernel
