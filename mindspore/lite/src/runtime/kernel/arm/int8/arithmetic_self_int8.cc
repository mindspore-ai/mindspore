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

#include "src/runtime/kernel/arm/int8/arithmetic_self_int8.h"
#include <limits>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ArithmeticSelfInt8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  int ret = ReSize();
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->GetQuantParams();
  para_->quant_arg_.in_args_.scale_ = in_quant_args.front().scale;
  para_->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint * (-1);

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->GetQuantParams();
  para_->quant_arg_.out_args_.scale_ = out_quant_args.front().scale;
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

  return ret;
}

int ArithmeticSelfInt8CPUKernel::ReSize() {
  data_size_ = in_tensors_[0]->ElementsNum();
  thread_sz_count_ = MSMIN(thread_count_, data_size_);
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int ArithmeticSelfInt8Runs(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<ArithmeticSelfInt8CPUKernel *>(cdata);
  auto ret = g_kernel->DoArithmeticSelf(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRuns error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ArithmeticSelfInt8CPUKernel::DoArithmeticSelf(int task_id) {
  int size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
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
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input_tensor = in_tensors_.at(0);
  auto out_tensor = out_tensors_.at(0);
  in_ptr_ = reinterpret_cast<int8_t *>(input_tensor->Data());
  out_ptr_ = reinterpret_cast<int8_t *>(out_tensor->Data());
  ret = LiteBackendParallelLaunch(ArithmeticSelfInt8Runs, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticSelfInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Creator failed, opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ArithmeticSelfInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Round, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Floor, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Ceil, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Abs, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Sin, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Cos, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Log, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Sqrt, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Rsqrt, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Square, CpuArithmeticSelfInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_LogicalNot, CpuArithmeticSelfInt8KernelCreator)
}  // namespace mindspore::kernel
