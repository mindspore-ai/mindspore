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

#include "src/litert/kernel/cpu/int8/split_int8.h"
#include <limits>
#include "nnacl/split_parameter.h"
#include "nnacl/int8/split_int8.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {
int SplitInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto ret = SplitBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output0 data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }

  output_ptr_.resize(param->num_split_);

  auto in_tensor = in_tensors_.at(kInputIndex);

  auto in_quant_args = in_tensor->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  param->quant_arg_.in_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  param->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;
  MS_CHECK_TRUE_RET(static_cast<size_t>(param->num_split_) == this->out_tensors_.size(), RET_ERROR);
  for (int i = 0; i < param->num_split_; i++) {
    auto *out_tensor = out_tensors_.at(i);
    auto out_quant_args = out_tensor->quant_params();
    CHECK_LESS_RETURN(out_quant_args.size(), 1);
    param->quant_arg_.out_args_[i].scale_ = static_cast<float>(out_quant_args.front().scale);
    param->quant_arg_.out_args_[i].zp_ = out_quant_args.front().zeroPoint;
  }

  param->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SplitInt8CPUKernel::ReSize() { return SplitBaseCPUKernel::ReSize(); }

int SplitInt8CPUKernel::Split(int task_id) {
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(task_id, thread_n_stride_), RET_ERROR);
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }

  int thread_offset = task_id * thread_n_stride_;
  CHECK_NULL_RETURN(input_ptr_);
  CHECK_NULL_RETURN(param);
  auto ret = Int8DoSplit(input_ptr_, output_ptr_.data(), in_tensors_.front()->shape().data(), thread_offset,
                         num_unit_thread, param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitInt8Run(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<SplitInt8CPUKernel *>(cdata);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitInt8CPUKernel::Run() {
  auto in_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<int8_t *>(in_tensor->MutableData());
  MS_CHECK_TRUE_RET(static_cast<size_t>(param->num_split_) == this->out_tensors_.size(), RET_ERROR);
  CHECK_LESS_RETURN(static_cast<int>(output_ptr_.size()), param->num_split_);
  for (int i = 0; i < param->num_split_; i++) {
    CHECK_NULL_RETURN(out_tensors_.at(i)->data());
    output_ptr_[i] = reinterpret_cast<int8_t *>(out_tensors_.at(i)->data());
  }

  auto ret = ParallelLaunch(this->ms_context_, SplitInt8Run, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Split, LiteKernelCreator<SplitInt8CPUKernel>)
}  // namespace mindspore::kernel
