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
#include "src/litert/kernel/cpu/int8/layer_norm_int8.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::kernel {
namespace {
constexpr int min_layernorm_input = 3;
constexpr int min_layernorm_output = 1;
}  // namespace
LayerNormInt8CPUKernel::~LayerNormInt8CPUKernel() {
  if (quant_param_ != nullptr) {
    free(quant_param_);
    quant_param_ = nullptr;
  }
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
  CHECK_NULL_RETURN(input);
  lite::Tensor *output = out_tensors_.at(0);
  CHECK_NULL_RETURN(output);

  const auto &input_params = input->quant_params();
  const auto &output_params = output->quant_params();
  MS_CHECK_TRUE_MSG(!input_params.empty(), RET_ERROR, "Input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!output_params.empty(), RET_ERROR, "Output quant param cannot be empty.");
  quant_param_ = reinterpret_cast<LayerNormQuantArg *>(malloc(sizeof(LayerNormQuantArg)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc LayerNormQuantArg for LayerNorm int8 op failed!";
    return RET_ERROR;
  }
  quant_param_->in_zp_ = input_params.front().zeroPoint;
  quant_param_->in_scale_ = input_params.front().scale;

  quant_param_->out_zp_ = output_params.front().zeroPoint;
  quant_param_->out_scale_ = output_params.front().scale;

  lite::Tensor *gamma_tensor = in_tensors_.at(1);
  CHECK_NULL_RETURN(gamma_tensor);
  if (gamma_tensor->quant_params().size() < 1) {
    MS_LOG(ERROR) << "LayerNorm int8 op gamma tensor error.";
    return RET_ERROR;
  }
  double gamma_scale = gamma_tensor->quant_params().front().scale;
  int gamma_zp = gamma_tensor->quant_params().front().zeroPoint;
  MS_CHECK_GT(gamma_tensor->ElementsNum(), 0, RET_ERROR);
  gamma_ptr_ = reinterpret_cast<float *>(malloc(gamma_tensor->ElementsNum() * sizeof(float)));
  CHECK_NULL_RETURN(gamma_ptr_);
  int8_t *src_gamma = reinterpret_cast<int8_t *>(gamma_tensor->data());
  for (int i = 0; i < gamma_tensor->ElementsNum(); i++) {
    gamma_ptr_[i] = (src_gamma[i] - gamma_zp) * gamma_scale;
  }

  lite::Tensor *beta_tensor = in_tensors_.at(2);
  CHECK_NULL_RETURN(beta_tensor);
  MS_CHECK_GT(beta_tensor->ElementsNum(), 0, RET_ERROR);
  beta_ptr_ = reinterpret_cast<float *>(malloc(beta_tensor->ElementsNum() * sizeof(float)));
  if (beta_ptr_ == nullptr) {
    MS_LOG(ERROR) << "malloc beta_ptr_ failed";
    free(gamma_ptr_);
    gamma_ptr_ = nullptr;
    return RET_ERROR;
  }
  int32_t *src_beta = reinterpret_cast<int32_t *>(beta_tensor->data());
  for (int i = 0; i < beta_tensor->ElementsNum(); i++) {
    beta_ptr_[i] = src_beta[i] * quant_param_->in_scale_ * gamma_scale;
  }
  return RET_OK;
}

int LayerNormInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), min_layernorm_input);
  CHECK_LESS_RETURN(out_tensors_.size(), min_layernorm_output);
  CHECK_NULL_RETURN(param_);

  auto ret = SetQuantArgs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set QuantArgs failed.";
    return ret;
  }

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
  param_->op_parameter_.thread_num_ = MSMIN(param_->norm_outer_size_, op_parameter_->thread_num_);
  return RET_OK;
}

int LayerNormInt8CPUKernel::DoExecute(int task_id) {
  auto ret = LayerNormInt8(src_ptr_, gamma_ptr_, beta_ptr_, dst_ptr_, param_, quant_param_, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoExecute task id " << task_id << " failed.";
    return ret;
  }
  return RET_OK;
}

int LayerNormInt8Run(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<LayerNormInt8CPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormInt8Run task_id " << task_id << " failed.";
    return ret;
  }
  return RET_OK;
}

int LayerNormInt8CPUKernel::Run() {
  src_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(src_ptr_);
  dst_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(dst_ptr_);

  auto ret = ParallelLaunch(this->ms_context_, LayerNormInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormInt8Run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_LayerNormFusion, LiteKernelCreator<LayerNormInt8CPUKernel>)
}  // namespace mindspore::kernel
