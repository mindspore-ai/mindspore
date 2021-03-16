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
#include "src/runtime/kernel/arm/fp32/layer_norm_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::kernel {
int LayerNormCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LayerNormCPUKernel::ReSize() {
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

int LayerNormCPUKernel::DoLayerNorm(int thread_id) {
  int ret = LayerNorm(src_data_, gamma_data_, beta_data_, dst_data_, mean_data_, var_data_, param_, thread_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoLayerNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int LayerNormRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<LayerNormCPUKernel *>(cdata);
  auto ret = kernel->DoLayerNorm(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LayerNormCPUKernel::Run() {
  int ret = RET_OK;
  src_data_ = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  gamma_data_ = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
  beta_data_ = reinterpret_cast<float *>(in_tensors_.at(2)->data_c());
  dst_data_ = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());
  if (out_tensors_.size() == 3) {
    mean_data_ = reinterpret_cast<float *>(out_tensors_.at(1)->data_c());
    var_data_ = reinterpret_cast<float *>(out_tensors_.at(2)->data_c());
  } else {
    mean_data_ = reinterpret_cast<float *>(context_->allocator->Malloc(param_->norm_outer_size_ * sizeof(float)));
    var_data_ = reinterpret_cast<float *>(context_->allocator->Malloc(param_->norm_outer_size_ * sizeof(float)));
  }
  ret = ParallelLaunch(this->context_->thread_pool_, LayerNormRun, this, op_parameter_->thread_num_);
  if (out_tensors_.size() != 3) {
    context_->allocator->Free(mean_data_);
    context_->allocator->Free(var_data_);
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LayerNormFusion, LiteKernelCreator<LayerNormCPUKernel>)
}  // namespace mindspore::kernel
