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

#include "src/litert/kernel/cpu/int8/arithmetic_self_int8.h"
#include <limits>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ArithmeticSelfInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputIndex + 1);
  CHECK_LESS_RETURN(out_tensors_.size(), kOutputIndex + 1);
  auto *input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  auto in_quant_args = input_tensor->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  para_->quant_arg_.in_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  para_->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint * (-1);

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  para_->quant_arg_.out_args_.scale_ = static_cast<float>(out_quant_args.front().scale);
  para_->quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  para_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  para_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();

  if (para_->op_parameter_.type_ == PrimitiveType_Square) {
    const double real_multiplier =
      (para_->quant_arg_.in_args_.scale_ * para_->quant_arg_.in_args_.scale_) / para_->quant_arg_.out_args_.scale_;

    int right_shift = 0;
    QuantizeMultiplierSmallerThanOne(real_multiplier, &para_->quant_arg_.output_multiplier_, &right_shift);

    para_->quant_arg_.shift_left_ = right_shift < 0 ? -right_shift : 0;
    para_->quant_arg_.shift_right_ = right_shift > 0 ? right_shift : 0;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticSelfInt8CPUKernel::ReSize() {
  data_size_ = in_tensors_[0]->ElementsNum();
  MS_CHECK_GT(data_size_, 0, RET_ERROR);
  thread_sz_count_ = MSMIN(thread_count_, static_cast<int>(data_size_));
  if (thread_sz_count_ == 0) {
    MS_LOG(ERROR) << "div zero";
    return RET_ERROR;
  }
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int ArithmeticSelfInt8Runs(void *cdata, int task_id, float, float) {
  auto g_kernel = reinterpret_cast<ArithmeticSelfInt8CPUKernel *>(cdata);
  auto ret = g_kernel->DoArithmeticSelf(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRuns error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ArithmeticSelfInt8CPUKernel::DoArithmeticSelf(int task_id) {
  int size = MSMIN(thread_sz_stride_, static_cast<int>(data_size_ - task_id * thread_sz_stride_));
  if (size <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  if (arithmeticSelf_run_) {
    auto ret = arithmeticSelf_run_(in_ptr_ + offset, out_ptr_ + offset, size, para_->quant_arg_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run failed, illegal input! ";
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "Run function is null! ";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfInt8CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto out_tensor = out_tensors_.at(0);
  in_ptr_ = reinterpret_cast<int8_t *>(input_tensor->MutableData());
  out_ptr_ = reinterpret_cast<int8_t *>(out_tensor->MutableData());
  auto ret = ParallelLaunch(this->ms_context_, ArithmeticSelfInt8Runs, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Round, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Floor, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Ceil, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Abs, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Sin, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Cos, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Log, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Sqrt, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Rsqrt, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Square, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_LogicalNot, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Reciprocal, LiteKernelCreator<ArithmeticSelfInt8CPUKernel>)
}  // namespace mindspore::kernel
