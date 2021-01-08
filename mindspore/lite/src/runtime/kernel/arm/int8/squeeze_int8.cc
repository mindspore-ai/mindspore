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

#include "src/runtime/kernel/arm/int8/squeeze_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore::kernel {
SqueezeInt8CPUKernel::~SqueezeInt8CPUKernel() {
  if (quant_squeeze_param_ != nullptr) {
    if (quant_squeeze_param_->in_quant_args_ != nullptr) {
      free(quant_squeeze_param_->in_quant_args_);
    }
    if (quant_squeeze_param_->out_quant_args_ != nullptr) {
      free(quant_squeeze_param_->out_quant_args_);
    }
    delete (quant_squeeze_param_);
  }
}

int SqueezeInt8CPUKernel::Init() {
  quant_squeeze_param_ = new (std::nothrow) SqueezeQuantArg;
  if (quant_squeeze_param_ == nullptr) {
    MS_LOG(ERROR) << "new quant_squeeze_param_ failed.";
    return RET_ERROR;
  }

  quant_squeeze_param_->in_quant_args_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (quant_squeeze_param_->in_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_squeeze_param_->in_quant_args_.";
    if (quant_squeeze_param_ != nullptr) {
      delete (quant_squeeze_param_);
      quant_squeeze_param_ = nullptr;
    }
    return RET_ERROR;
  }
  auto in_quant_args = in_tensors_.front()->quant_params();
  MS_ASSERT(in_quant_args.size() > 0);
  quant_squeeze_param_->in_quant_args_->scale_ = in_quant_args.front().scale;
  quant_squeeze_param_->in_quant_args_->zp_ = in_quant_args.front().zeroPoint;

  MS_ASSERT(this->out_tensors_.size() == 1);
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(output_tensor != nullptr);
  auto quant_args = output_tensor->quant_params();
  MS_ASSERT(quant_args.size() == 1);
  quant_squeeze_param_->out_quant_args_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (quant_squeeze_param_->in_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc QuantArg failed";
    if (quant_squeeze_param_ != nullptr) {
      if (quant_squeeze_param_->in_quant_args_ != nullptr) {
        free(quant_squeeze_param_->in_quant_args_);
        quant_squeeze_param_->in_quant_args_ = nullptr;
      }
      delete (quant_squeeze_param_);
      quant_squeeze_param_ = nullptr;
    }
    return RET_ERROR;
  }
  quant_squeeze_param_->out_quant_args_->scale_ = quant_args.front().scale;
  quant_squeeze_param_->out_quant_args_->zp_ = quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SqueezeInt8CPUKernel::ReSize() { return RET_OK; }

int SqueezeInt8CPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, SqueezeInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunSqueezeParam failed. errorcode: ";
  }
  return ret;
}

int SqueezeInt8Run(void *cdata, int task_id) {
  auto Squeeze = reinterpret_cast<SqueezeInt8CPUKernel *>(cdata);
  Squeeze->DoExecute(task_id);
  return RET_OK;
}

int SqueezeInt8CPUKernel::DoExecute(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  MS_ASSERT(input_tensor);
  auto out_tensor = out_tensors_.at(kOutputIndex);
  MS_ASSERT(out_tensor);
  int8_t *input_data = reinterpret_cast<int8_t *>(input_tensor->data_c());
  MS_ASSERT(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensor->data_c());
  MS_ASSERT(output_data);

  int num = input_tensor->ElementsNum();
  SqueezeInt8(input_data, output_data, quant_squeeze_param_, num, task_id, op_parameter_->thread_num_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Squeeze, LiteKernelCreator<SqueezeInt8CPUKernel>)
}  // namespace mindspore::kernel
