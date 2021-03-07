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
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::kernel {
int PoolingBaseCPUKernel::SetQuantParam() {
  // per tensor init
  pooling_quant_arg_ = reinterpret_cast<QuantArg **>(malloc(2 * sizeof(QuantArg *)));
  if (pooling_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "malloc pooling_quant_arg failed.";
    return RET_MEMORY_FAILED;
  }
  pooling_quant_arg_[0] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (pooling_quant_arg_[0] == nullptr) {
    pooling_quant_arg_[1] = nullptr;
    MS_LOG(ERROR) << "malloc pooling_quant_arg[0] failed.";
    free(pooling_quant_arg_);
    pooling_quant_arg_ = nullptr;
    return RET_MEMORY_FAILED;
  }
  pooling_quant_arg_[1] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (pooling_quant_arg_[1] == nullptr) {
    MS_LOG(ERROR) << "malloc pooling_quant_arg[1] failed.";
    free(pooling_quant_arg_[0]);
    pooling_quant_arg_[0] = nullptr;
    free(pooling_quant_arg_);
    pooling_quant_arg_ = nullptr;
    return RET_MEMORY_FAILED;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_arg = input_tensor->quant_params();
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_arg = out_tensor->quant_params();
  pooling_quant_arg_[0][0].scale_ = in_quant_arg.front().scale;
  pooling_quant_arg_[0][0].zp_ = in_quant_arg.front().zeroPoint;
  pooling_quant_arg_[1][0].scale_ = out_quant_arg.front().scale;
  pooling_quant_arg_[1][0].zp_ = out_quant_arg.front().zeroPoint;
  pooling_param_->quant_args_ = pooling_quant_arg_;
  if (pooling_quant_arg_[0][0].scale_ == pooling_quant_arg_[1][0].scale_ &&
      pooling_quant_arg_[0][0].zp_ == pooling_quant_arg_[1][0].zp_) {
    pooling_param_->quantize_ = false;
  } else {
    pooling_param_->quantize_ = true;
  }
  return RET_OK;
}

void PoolingBaseCPUKernel::FreeQuantParam() {
  if (pooling_quant_arg_ != nullptr) {
    for (int i = 0; i < 2; ++i) {
      if (*(pooling_quant_arg_ + i) != nullptr) {
        free(*(pooling_quant_arg_ + i));
      }
    }
    free(pooling_quant_arg_);
    pooling_quant_arg_ = nullptr;
  }
}

int PoolingBaseCPUKernel::Init() {
  MS_ASSERT(in_tensors_.size() == 1);
  MS_ASSERT(out_tensors_.size() == 1);
  pooling_param_->thread_num_ = thread_count_;
  MS_ASSERT(this->op_parameter_ != nullptr);
  return RET_OK;
}

int PoolingBaseCPUKernel::ReSize() {
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
}  // namespace mindspore::kernel
