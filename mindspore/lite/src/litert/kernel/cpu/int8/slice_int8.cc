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

#include "src/litert/kernel/cpu/int8/slice_int8.h"
#include <limits>
#include "src/litert/kernel_registry.h"
#include "nnacl/int8/slice_int8.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore::kernel {
int SliceInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);
  CHECK_NULL_RETURN(param_);

  auto in_quant_args = input->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_ERROR, "Input quant param cannot be empty.");
  param_->quant_arg_.in_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  param_->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto out_quant_args = output->quant_params();
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_ERROR, "Output quant param cannot be empty.");
  param_->quant_arg_.out_args_.scale_ = static_cast<float>(out_quant_args.front().scale);
  param_->quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  QuantizeRoundParameterWithDoublePrecision(param_->quant_arg_.in_args_.scale_ / param_->quant_arg_.out_args_.scale_,
                                            &param_->quant_arg_.multiplier_.multiplier_,
                                            &param_->quant_arg_.multiplier_.left_shift_,
                                            &param_->quant_arg_.multiplier_.right_shift_);

  param_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SliceInt8CPUKernel::DoSlice(int task_id) {
  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(output_data);

  auto ret = SliceInt8(input_data, output_data, param_, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SliceInt8 error ,task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SliceInt8Run(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto slice_kernel = reinterpret_cast<SliceInt8CPUKernel *>(cdata);
  auto ret = slice_kernel->DoSlice(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoSlice error, task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SliceInt8CPUKernel::Run() {
  // param_ shape info has already been extended to 8d
  auto ret = ParallelLaunch(this->ms_context_, SliceInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SliceInt8Run error, error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_SliceFusion, LiteKernelCreator<SliceInt8CPUKernel>)
}  // namespace mindspore::kernel
