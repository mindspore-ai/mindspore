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

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::kernel {
LayerNormInt8CPUKernel::~LayerNormInt8CPUKernel() {
  if (gamma_ptr_ != nullptr) {
    free(gamma_ptr_);
    gamma_ptr_ = nullptr;
  }
  if (beta_ptr_ != nullptr) {
    free(beta_ptr_);
    beta_ptr_ = nullptr;
  }
}

int LayerNormInt8CPUKernel::SetQuantArgs() {
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);

  quant_param_.in_zp_ = input->quant_params().front().zeroPoint;
  quant_param_.in_scale_ = input->quant_params().front().scale;
  quant_param_.out_zp_ = output->quant_params().front().zeroPoint;
  quant_param_.out_scale_ = output->quant_params().front().scale;

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
  auto shape = in_tensors_.front()->shape();
  param_->begin_norm_axis_ =
    param_->begin_norm_axis_ > 0 ? param_->begin_norm_axis_ : param_->begin_norm_axis_ + shape.size();
  param_->begin_params_axis_ =
    param_->begin_params_axis_ > 0 ? param_->begin_params_axis_ : param_->begin_params_axis_ + shape.size();

  param_->norm_outer_size_ = 1;
  for (int i = 0; i < param_->begin_norm_axis_; ++i) {
    param_->norm_outer_size_ *= shape.at(i);
  }
  param_->norm_inner_size_ = 1;
  for (size_t i = param_->begin_norm_axis_; i < shape.size(); ++i) {
    param_->norm_inner_size_ *= shape.at(i);
  }
  param_->params_outer_size_ = 1;
  for (int i = 0; i < param_->begin_params_axis_; ++i) {
    param_->params_outer_size_ *= shape.at(i);
  }
  param_->params_inner_size_ = 1;
  for (size_t i = param_->begin_params_axis_; i < shape.size(); ++i) {
    param_->params_inner_size_ *= shape.at(i);
  }
  param_->op_parameter_.thread_num_ = MSMIN(param_->norm_outer_size_, context_->thread_num_);
  return RET_OK;
}

int LayerNormInt8CPUKernel::DoExecute(int task_id) {
  LayerNormInt8(src_ptr_, gamma_ptr_, beta_ptr_, dst_ptr_, param_, &quant_param_, task_id);
  return RET_OK;
}

int LayerNormInt8Run(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<LayerNormInt8CPUKernel *>(cdata);
  kernel->DoExecute(task_id);
  return RET_OK;
}

int LayerNormInt8CPUKernel::Run() {
  src_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data_c());
  dst_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data_c());

  auto ret = ParallelLaunch(this->context_->thread_pool_, LayerNormInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormInt8Run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_LayerNormFusion, LiteKernelCreator<LayerNormInt8CPUKernel>)
}  // namespace mindspore::kernel
