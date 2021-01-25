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
#include "src/runtime/kernel/arm/int8/l2_norm_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_L2NormalizeFusion;

namespace mindspore::kernel {
int L2NormInt8CPUKernel::Init() {
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  quant_param_.in_.scale_ = input->quant_params().front().scale;
  quant_param_.in_.zp_ = input->quant_params().front().zeroPoint;
  quant_param_.out_.scale_ = output->quant_params().front().scale;
  quant_param_.out_.zp_ = output->quant_params().front().zeroPoint;
  return ReSize();
}

int L2NormInt8Run(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<L2NormInt8CPUKernel *>(cdata);
  kernel->DoExecute(task_id);
  return lite::RET_OK;
}

int L2NormInt8CPUKernel::Run() {
  if (l2_norm_param_->axis_num_ != 1 || l2_norm_param_->axis_[0] != static_cast<int>(l2_norm_param_->shape_num_) - 1) {
    MS_LOG(ERROR) << "L2Norm only support reduce on all axis and trailing axis with trailing axis";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(context_->thread_pool_, L2NormInt8Run, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "L2Norm error: error_code[" << ret << "]";
  }
  return ret;
}

int L2NormInt8CPUKernel::DoExecute(int task_id) {
  lite::Tensor *input_tensor = in_tensors().front();
  int outer_size = input_tensor->ElementsNum() / input_tensor->shape().back();
  int stride = UP_DIV(outer_size, context_->thread_num_);
  int begin = task_id * stride;
  int end = MSMIN(begin + stride, outer_size);

  int8_t *input_data = static_cast<int8_t *>(in_tensors().front()->MutableData());
  MS_ASSERT(input_data);
  int8_t *output_data = static_cast<int8_t *>(out_tensors().front()->MutableData());
  MS_ASSERT(output_data);
  MS_ASSERT(l2_norm_param_);
  return L2NormalizationInt8(input_data, output_data, l2_norm_param_, &quant_param_, begin, end);
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_L2NormalizeFusion, LiteKernelCreator<L2NormInt8CPUKernel>)
}  // namespace mindspore::kernel
