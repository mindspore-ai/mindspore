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

#include "src/litert/kernel/cpu/int8/relux_int8.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_RELU;

namespace mindspore::kernel {
int ReluXInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);
  const auto &input_params = input->quant_params();
  const auto &output_params = output->quant_params();
  MS_CHECK_TRUE_MSG(!input_params.empty(), RET_ERROR, "Input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!output_params.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_arg_.input_arg.scale_ = static_cast<float>(input_params.front().scale);
  quant_arg_.input_arg.zp_ = input_params.front().zeroPoint;
  quant_arg_.output_arg.scale_ = static_cast<float>(output_params.front().scale);
  quant_arg_.output_arg.zp_ = output_params.front().zeroPoint;

  const double multiplier = quant_arg_.input_arg.scale_ / quant_arg_.output_arg.scale_;
  QuantizeRoundParameterWithDoublePrecision(multiplier, &quant_arg_.input_multiplier_, &quant_arg_.left_shift_,
                                            &quant_arg_.right_shift_);

  return RET_OK;
}

int ReluXInt8CPUKernel::ReSize() { return RET_OK; }

int ReluXInt8CPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input_addr);
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_addr);
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, op_parameter_->thread_num_);
  int count = MSMIN(stride, length - stride * task_id);

  ReluXInt8(input_addr + stride * task_id, count, output_addr + stride * task_id, &quant_arg_);
  return RET_OK;
}

int ReluXInt8Run(void *cdata, int task_id, float, float) {
  auto activation_kernel = reinterpret_cast<ReluXInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ReluXInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReluXInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ReluXInt8Run, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ReluXInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
