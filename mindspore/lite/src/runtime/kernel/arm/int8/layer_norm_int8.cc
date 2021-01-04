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
#include "src/runtime/kernel/arm/int8/layer_norm_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/ops/populate/layer_norm_populate.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNorm;

namespace mindspore::kernel {
LayerNormInt8CPUKernel::~LayerNormInt8CPUKernel() {
  if (param_->elementwise_mode_ != 0 && gamma_ptr_ != nullptr) {
    free(gamma_ptr_);
    gamma_ptr_ = nullptr;
  }
  if (param_->elementwise_mode_ != 0 && beta_ptr_ != nullptr) {
    free(beta_ptr_);
    beta_ptr_ = nullptr;
  }
  return;
}

int LayerNormInt8CPUKernel::SetQuantArgs() {
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);

  quant_param_.in_zp_ = input->quant_params().front().zeroPoint;
  quant_param_.in_scale_ = input->quant_params().front().scale;
  quant_param_.out_zp_ = output->quant_params().front().zeroPoint;
  quant_param_.out_scale_ = output->quant_params().front().scale;

  if (param_->elementwise_mode_ != 0) {
    lite::Tensor *gamma_tensor = in_tensors_.at(1);
    lite::Tensor *beta_tensor = in_tensors_.at(2);

    double gamma_scale = gamma_tensor->quant_params().front().scale;
    int gamma_zp = gamma_tensor->quant_params().front().zeroPoint;
    gamma_ptr_ = reinterpret_cast<float *>(malloc(gamma_tensor->ElementsNum() * sizeof(float)));
    if (gamma_ptr_ == nullptr) {
      MS_LOG(ERROR) << "malloc gamma_ptr_ failed";
      return RET_ERROR;
    }
    int8_t *src_gamma = reinterpret_cast<int8_t *>(gamma_tensor->data_c());
    for (int i = 0; i < gamma_tensor->ElementsNum(); i++) {
      gamma_ptr_[i] = (src_gamma[i] - gamma_zp) * gamma_scale;
    }

    beta_ptr_ = reinterpret_cast<float *>(malloc(beta_tensor->ElementsNum() * sizeof(float)));
    if (beta_ptr_ == nullptr) {
      MS_LOG(ERROR) << "malloc beta_ptr_ failed";
      free(gamma_ptr_);
      gamma_ptr_ = nullptr;
      return RET_ERROR;
    }
    int32_t *src_beta = reinterpret_cast<int32_t *>(beta_tensor->data_c());
    for (int i = 0; i < beta_tensor->ElementsNum(); i++) {
      beta_ptr_[i] = src_beta[i] * quant_param_.in_scale_ * gamma_scale;
    }
  }
  return RET_OK;
}

int LayerNormInt8CPUKernel::Init() {
  SetQuantArgs();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LayerNormInt8CPUKernel::ReSize() {
  if (op_parameter_ != nullptr) {
    free(op_parameter_);
    op_parameter_ = nullptr;
  }
  op_parameter_ = PopulateLayerNormParameter(primitive_);
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "op_parameter_ is nullptr!";
    return RET_NULL_PTR;
  }
  op_parameter_->thread_num_ = context_->thread_num_;
  param_ = reinterpret_cast<LayerNormParameter *>(op_parameter_);
  auto shape = in_tensors_.front()->shape();
  outer_size_ = 1;
  inner_size_ = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i + param_->normalized_dims_ < shape.size()) {
      outer_size_ *= shape.at(i);
    } else {
      inner_size_ *= shape.at(i);
    }
  }

  param_->thread_count_ = MSMIN(outer_size_, op_parameter_->thread_num_);
  param_->thread_outsize_ = UP_DIV(outer_size_, param_->thread_count_);
  return RET_OK;
}

int LayerNormInt8Run(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<LayerNormInt8CPUKernel *>(cdata);
  kernel->DoExecute(task_id);
  return RET_OK;
}

int LayerNormInt8CPUKernel::DoExecute(int task_id) {
  int current_out_size = outer_size_ - task_id * param_->thread_outsize_;
  current_out_size = MSMIN(current_out_size, param_->thread_outsize_);
  if (current_out_size <= 0) {
    return RET_OK;
  }

  const int8_t *thread_src = src_ptr_ + task_id * param_->thread_outsize_ * inner_size_;
  int8_t *thread_dst = dst_ptr_ + task_id * param_->thread_outsize_ * inner_size_;

  LayerNormInt8(thread_src, gamma_ptr_, beta_ptr_, thread_dst, param_->elementwise_mode_, current_out_size, inner_size_,
                &quant_param_, param_->epsilon_);
  return RET_OK;
}

int LayerNormInt8CPUKernel::Run() {
  src_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  dst_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());

  auto ret = ParallelLaunch(this->context_->thread_pool_, LayerNormInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormInt8Run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_LayerNorm, LiteKernelCreator<LayerNormInt8CPUKernel>)
}  // namespace mindspore::kernel
