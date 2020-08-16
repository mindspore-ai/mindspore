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
#include "src/runtime/kernel/arm/base/pooling_base.h"
#include <vector>
#include "src/runtime/kernel/arm/int8/pooling_int8.h"
#include "src/runtime/kernel/arm/fp32/pooling.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pooling;

namespace mindspore::kernel {
int PoolingBaseCPUKernel::SetQuantParam() {
  // per tensor init
  pooling_quant_arg_ = reinterpret_cast<QuantArg **>(malloc(2 * sizeof(QuantArg *)));
  pooling_quant_arg_[0] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  pooling_quant_arg_[1] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_arg = input_tensor->GetQuantParams();
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_arg = out_tensor->GetQuantParams();
  pooling_quant_arg_[0][0].scale_ = in_quant_arg.front().scale;
  pooling_quant_arg_[0][0].zp_ = in_quant_arg.front().zeroPoint;
  pooling_quant_arg_[1][0].scale_ = out_quant_arg.front().scale;
  pooling_quant_arg_[1][0].zp_ = out_quant_arg.front().zeroPoint;
  pooling_param_->quant_args_ = pooling_quant_arg_;
  return RET_OK;
}

void PoolingBaseCPUKernel::FreeQuantParam() {
  if (pooling_quant_arg_ != nullptr) {
    for (int i = 0; i < 2; ++i) {
      if (*(pooling_quant_arg_ + i) != nullptr) {
        free(*(pooling_quant_arg_ + i));
      }
    }
  }
}

int PoolingBaseCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  MS_ASSERT(in_tensors_.size() == 1);
  MS_ASSERT(out_tensors_.size() == 1);
  pooling_param_->thread_num_ = thread_count_;
  MS_ASSERT(this->op_parameter_ != nullptr);
  auto in_tensor = this->in_tensors_.front();
  auto out_tensor = this->out_tensors_.front();
  MS_ASSERT(in_tensor != nullptr);
  MS_ASSERT(out_tensor != nullptr);
  pooling_param_->input_batch_ = in_tensor->Batch();
  pooling_param_->input_channel_ = in_tensor->Channel();
  pooling_param_->input_h_ = in_tensor->Height();
  pooling_param_->input_w_ = in_tensor->Width();
  pooling_param_->output_batch_ = out_tensor->Batch();
  pooling_param_->output_channel_ = out_tensor->Channel();
  pooling_param_->output_h_ = out_tensor->Height();
  pooling_param_->output_w_ = out_tensor->Width();
  if (pooling_param_->global_) {
      pooling_param_->window_h_ = pooling_param_->input_h_;
      pooling_param_->window_w_ = pooling_param_->input_w_;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPoolingInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const Context *ctx,
                                                const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Pooling);
  auto *kernel = new (std::nothrow) PoolingInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PoolingInt8CPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuPoolingFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const Context *ctx,
                                                const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Pooling);
  auto *kernel = new (std::nothrow) PoolingCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PoolingCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Pooling, CpuPoolingInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Pooling, CpuPoolingFp32KernelCreator)
}  // namespace mindspore::kernel
