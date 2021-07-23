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
#include "src/runtime/kernel/arm/fp16/layer_norm_fp16.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/layer_norm_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::kernel {
int LayerNormFp16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LayerNormFp16CPUKernel::ReSize() {
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
  op_parameter_->thread_num_ = MSMIN(param_->norm_outer_size_, op_parameter_->thread_num_);
  return RET_OK;
}

int LayerNormFp16CPUKernel::DoLayerNormFp16(int thread_id) {
  int ret = LayerNormFp16(src_data_, gamma_data_, beta_data_, dst_data_, mean_data_, var_data_, param_, thread_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoLayerNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int LayerNormFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<LayerNormFp16CPUKernel *>(cdata);
  auto ret = kernel->DoLayerNormFp16(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormFp16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LayerNormFp16CPUKernel::Run() {
  int ret = RET_OK;
  src_data_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c());
  gamma_data_ = reinterpret_cast<float16_t *>(in_tensors_.at(GAMMA_INDEX)->data_c());
  beta_data_ = reinterpret_cast<float16_t *>(in_tensors_.at(BETA_INDEX)->data_c());
  dst_data_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data_c());
  if (out_tensors_.size() == kInputSize2) {
    mean_data_ = reinterpret_cast<float16_t *>(out_tensors_.at(MEAN_INDEX)->data_c());
    var_data_ = reinterpret_cast<float16_t *>(out_tensors_.at(VARIANCE_INDEX)->data_c());
  } else {
    mean_data_ =
      reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(param_->norm_outer_size_ * sizeof(float16_t)));
    var_data_ =
      reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(param_->norm_outer_size_ * sizeof(float16_t)));
  }
  ret = ParallelLaunch(this->ms_context_, LayerNormFp16Run, this, op_parameter_->thread_num_);
  if (out_tensors_.size() != kInputSize2) {
    ms_context_->allocator->Free(mean_data_);
    ms_context_->allocator->Free(var_data_);
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LayerNormFusion, LiteKernelCreator<LayerNormFp16CPUKernel>)
}  // namespace mindspore::kernel
