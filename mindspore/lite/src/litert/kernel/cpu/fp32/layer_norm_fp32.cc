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
#include "src/litert/kernel/cpu/fp32/layer_norm_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::kernel {
int LayerNormCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputSize2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(param_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LayerNormCPUKernel::ReSize() {
  auto input = in_tensors_.front();
  CHECK_NULL_RETURN(input);
  auto shape = input->shape();
  param_->begin_norm_axis_ =
    param_->begin_norm_axis_ >= 0 ? param_->begin_norm_axis_ : param_->begin_norm_axis_ + shape.size();
  param_->begin_params_axis_ =
    param_->begin_params_axis_ >= 0 ? param_->begin_params_axis_ : param_->begin_params_axis_ + shape.size();
  MS_CHECK_LT(param_->begin_norm_axis_, static_cast<int>(shape.size()), RET_ERROR);
  MS_CHECK_LT(param_->begin_params_axis_, static_cast<int>(shape.size()), RET_ERROR);
  param_->norm_outer_size_ = 1;
  for (int i = 0; i < param_->begin_norm_axis_; ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->norm_outer_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->norm_outer_size_ *= shape.at(i);
  }
  param_->norm_inner_size_ = 1;
  for (size_t i = param_->begin_norm_axis_; i < shape.size(); ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->norm_inner_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->norm_inner_size_ *= shape.at(i);
  }
  param_->params_outer_size_ = 1;
  for (int i = 0; i < param_->begin_params_axis_; ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->params_outer_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->params_outer_size_ *= shape.at(i);
  }
  param_->params_inner_size_ = 1;
  for (size_t i = param_->begin_params_axis_; i < shape.size(); ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->params_inner_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->params_inner_size_ *= shape.at(i);
  }
  if (UpdateThreadNumPass(TC_PTYPE(PrimitiveType_LayerNormFusion), param_->norm_inner_size_, param_->norm_inner_size_,
                          out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    return RET_ERROR;
  }
  thread_num_ = MSMIN(param_->norm_outer_size_, thread_num_);
  return RET_OK;
}

int LayerNormCPUKernel::DoLayerNorm(int thread_id) const {
  auto ret = LayerNorm(src_data_, gamma_data_, beta_data_, dst_data_, mean_data_, var_data_, param_, thread_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoLayerNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int LayerNormRun(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<const LayerNormCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->DoLayerNorm(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LayerNormCPUKernel::Run() {
  src_data_ = reinterpret_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
  CHECK_NULL_RETURN(src_data_);
  gamma_data_ = reinterpret_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
  CHECK_NULL_RETURN(gamma_data_);
  beta_data_ = reinterpret_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
  CHECK_NULL_RETURN(beta_data_);
  dst_data_ = reinterpret_cast<float *>(out_tensors_.at(FIRST_INPUT)->data());
  CHECK_NULL_RETURN(dst_data_);

  if (out_tensors_.size() == 3) {
    mean_data_ = reinterpret_cast<float *>(out_tensors_.at(SECOND_INPUT)->data());
    CHECK_NULL_RETURN(mean_data_);
    var_data_ = reinterpret_cast<float *>(out_tensors_.at(THIRD_INPUT)->data());
    CHECK_NULL_RETURN(var_data_);
  } else if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "LayerNorm should have 1 or 3 output tensors";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, LayerNormRun, this, thread_num_);
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LayerNormFusion, LiteKernelCreator<LayerNormCPUKernel>)
}  // namespace mindspore::kernel
