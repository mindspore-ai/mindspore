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
#include "src/litert/kernel/cpu/int8/l2_norm_int8.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_L2NormalizeFusion;

namespace mindspore::kernel {
L2NormInt8CPUKernel::~L2NormInt8CPUKernel() {
  if (quant_param_ != nullptr) {
    free(quant_param_);
    quant_param_ = nullptr;
  }
}

int L2NormInt8CPUKernel::Prepare() {
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
  MS_ASSERT(input);
  MS_ASSERT(output);

  quant_param_ = reinterpret_cast<L2NormQuantArg *>(malloc(sizeof(L2NormQuantArg)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc L2NormQuantArg for L2Norm int8 op failed!";
    return RET_ERROR;
  }
  const auto &input_params = input->quant_params();
  const auto &output_params = output->quant_params();
  MS_CHECK_TRUE_MSG(!input_params.empty(), RET_ERROR, "Input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!output_params.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_param_->in_.scale_ = static_cast<float>(input_params.front().scale);
  quant_param_->in_.zp_ = input_params.front().zeroPoint;
  quant_param_->out_.scale_ = static_cast<float>(output_params.front().scale);
  quant_param_->out_.zp_ = output_params.front().zeroPoint;
  return ReSize();
}

int L2NormInt8Run(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<L2NormInt8CPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "L2NormInt8Run task id " << task_id << " failed.";
    return ret;
  }
  return lite::RET_OK;
}

int L2NormInt8CPUKernel::Run() {
  if (l2_norm_param_->axis_num_ != 1 || l2_norm_param_->axis_[0] != static_cast<int>(l2_norm_param_->shape_num_) - 1) {
    MS_LOG(ERROR) << "L2Norm only support reduce on all axis and trailing axis with trailing axis";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, L2NormInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "L2Norm error: error_code[" << ret << "]";
  }
  return ret;
}

int L2NormInt8CPUKernel::DoExecute(int task_id) {
  auto input_tensor = static_cast<lite::Tensor *>(in_tensors().front());
  MS_CHECK_GT(input_tensor->shape().back(), 0, RET_ERROR);
  MS_CHECK_GT(input_tensor->ElementsNum(), 0, RET_ERROR);
  int outer_size = input_tensor->ElementsNum() / input_tensor->shape().back();
  int stride = UP_DIV(outer_size, op_parameter_->thread_num_);
  if (INT_MUL_OVERFLOW(task_id, stride)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int begin = task_id * stride;
  int end = MSMIN(begin + stride, outer_size);

  int8_t *input_data = static_cast<int8_t *>(in_tensors().front()->MutableData());
  CHECK_NULL_RETURN(input_data);
  int8_t *output_data = static_cast<int8_t *>(out_tensors().front()->MutableData());
  CHECK_NULL_RETURN(output_data);
  MS_ASSERT(l2_norm_param_);
  return L2NormalizationInt8(input_data, output_data, l2_norm_param_, quant_param_, begin, end);
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_L2NormalizeFusion, LiteKernelCreator<L2NormInt8CPUKernel>)
}  // namespace mindspore::kernel
