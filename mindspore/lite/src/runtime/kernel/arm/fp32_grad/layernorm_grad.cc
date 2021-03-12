/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32_grad/layernorm_grad.h"
#include <vector>

#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32_grad/layernorm_grad.h"
#include "nnacl/fp32_grad/layernormgrad_parameter.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNormGrad;

namespace mindspore::kernel {
int LayerNormGradCPUKernel::ReSize() { return RET_OK; }

int LayerNormGradCPUKernel::Init() {
  auto lngrad_param = reinterpret_cast<LayerNormGradParameter *>(op_parameter_);
  auto *input_x = in_tensors_.at(0);
  std::vector<int> x_shape = input_x->shape();
  int begin_norm_axis = lngrad_param->begin_norm_axis_;
  if (begin_norm_axis < 0) {
    begin_norm_axis += x_shape.size();
  }
  auto begin_params_axis = lngrad_param->begin_params_axis_;
  if (begin_params_axis < 0) {
    begin_params_axis += x_shape.size();
  }
  for (size_t i = 0; i < static_cast<size_t>(begin_norm_axis); i++) {
    block_num_ *= x_shape[i];
  }
  for (size_t i = static_cast<size_t>(begin_norm_axis); i < x_shape.size(); i++) {
    block_size_ *= x_shape[i];
  }
  for (size_t i = 0; i < static_cast<size_t>(begin_params_axis); i++) {
    param_size_ *= x_shape[i];
  }
  for (size_t i = begin_params_axis; i < x_shape.size(); i++) {
    param_num_ *= x_shape[i];
  }
  if (block_num_ <= 0 || block_size_ <= 0) {
    MS_LOG(ERROR) << "LayerNormGradCPUKernel input shape error, input shape: " << x_shape;
  }
  return RET_OK;
}

int LayerNormGradCPUKernel::Execute(int task_id) {
  auto input_x = in_tensors_.at(0);
  auto input_dy = in_tensors_.at(1);
  auto input_var = in_tensors_.at(2);
  auto input_mean = in_tensors_.at(3);
  auto input_gamma = in_tensors_.at(4);
  auto output_dx = out_tensors_.at(0);
  auto output_dg = out_tensors_.at(1);
  auto output_db = out_tensors_.at(2);

  float *x = reinterpret_cast<float *>(input_x->MutableData());
  float *dy = reinterpret_cast<float *>(input_dy->MutableData());
  float *var = reinterpret_cast<float *>(input_var->MutableData());
  float *mean = reinterpret_cast<float *>(input_mean->MutableData());
  float *gamma = reinterpret_cast<float *>(input_gamma->MutableData());
  float *dx = reinterpret_cast<float *>(output_dx->MutableData());
  float *dg = reinterpret_cast<float *>(output_dg->MutableData());
  float *db = reinterpret_cast<float *>(output_db->MutableData());
  LayerNormGrad(x, dy, var, mean, gamma, param_num_, param_size_, block_num_, block_size_, dx, dg, db);
  return RET_OK;
}

int LayerNormGradRun(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto ln_kernel = reinterpret_cast<LayerNormGradCPUKernel *>(cdata);
  auto error_code = ln_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "LayerNormGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LayerNormGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, LayerNormGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "LayerNorm function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LayerNormGrad, LiteKernelCreator<LayerNormGradCPUKernel>)
}  // namespace mindspore::kernel
