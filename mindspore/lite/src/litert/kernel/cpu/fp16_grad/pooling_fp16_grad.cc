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

#include "src/litert/kernel/cpu/fp16_grad/pooling_fp16_grad.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp16/pooling_fp16.h"
#include "nnacl/fp16_grad/pooling_grad.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AvgPoolGrad;
using mindspore::schema::PrimitiveType_MaxPoolGrad;

namespace mindspore::kernel {
namespace {
constexpr int kNumInputDim_2 = 2;
constexpr int kNumShapeDim_2 = 2;
}  // namespace
int PoolingGradCPUKernelFp16::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_1D);
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(op_parameter_);
  CHECK_NULL_RETURN(pool_param);
  CHECK_NULL_RETURN(in_tensors_.at(FIRST_INPUT));
  CHECK_NULL_RETURN(in_tensors_.at(SECOND_INPUT));
  CHECK_NULL_RETURN(in_tensors_.at(kNumInputDim_2));
  CHECK_NULL_RETURN(out_tensors_.at(FIRST_INPUT));

  auto in_shape = in_tensors_.at(FIRST_INPUT)->shape();
  auto out_shape = in_tensors_.at(SECOND_INPUT)->shape();
  MS_CHECK_TRUE_RET(in_shape.size() == COMM_SHAPE_SIZE, RET_ERROR);
  MS_CHECK_TRUE_RET(out_shape.size() == COMM_SHAPE_SIZE, RET_ERROR);

  if (pool_param->pool_mode_ == PoolMode_AvgPool) {
    out_shape = in_tensors_.at(kNumInputDim_2)->shape();
  }
  int input_h = in_shape.at(SECOND_INPUT);
  int input_w = in_shape.at(kNumShapeDim_2);
  MS_CHECK_TRUE_RET(input_h > 0, RET_ERROR);
  MS_CHECK_TRUE_RET(input_w > 0, RET_ERROR);

  compute_.window_w_ = pool_param->window_w_;
  compute_.window_h_ = pool_param->window_h_;
  if (pool_param->global_) {
    compute_.window_w_ = input_w;
    compute_.window_h_ = input_h;
  }

  compute_.input_h_ = in_shape[kNHWC_H];
  compute_.input_w_ = in_shape[kNHWC_W];
  compute_.input_batch_ = in_shape[kNHWC_N];
  compute_.input_channel_ = in_shape[kNHWC_C];
  compute_.output_h_ = out_shape[kNHWC_H];
  compute_.output_w_ = out_shape[kNHWC_W];
  compute_.output_batch_ = out_shape[kNHWC_N];
  compute_.output_channel_ = out_shape[kNHWC_C];
  return RET_OK;
}

int PoolingGradCPUKernelFp16::Prepare() { return ReSize(); }

int PoolingGradCPUKernelFp16::DoExecute(int task_id) {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(op_parameter_);
  auto input_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(output_ptr);
  MS_CHECK_TRUE_RET(thread_num_ > 0, RET_ERROR);
  int stride = UP_DIV(compute_.output_batch_, thread_num_);
  int count = MSMIN(stride, compute_.output_batch_ - stride * task_id);
  if (count > 0) {
    int in_batch_size = compute_.input_h_ * compute_.input_w_ * compute_.input_channel_;
    int out_batch_size = compute_.output_h_ * compute_.output_w_ * compute_.input_channel_;
    std::fill(output_ptr + task_id * stride * in_batch_size, output_ptr + ((task_id * stride) + count) * in_batch_size,
              0.f);
    if (pool_param->pool_mode_ == PoolMode_MaxPool) {
      auto dy_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(kNumInputDim_2)->data());
      CHECK_NULL_RETURN(dy_ptr);
      MaxPoolingFp16Grad(input_ptr + task_id * stride * in_batch_size, dy_ptr + task_id * stride * out_batch_size,
                         output_ptr + task_id * stride * in_batch_size, count, pool_param, &compute_);
    } else {
      input_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(kNumInputDim_2)->data());
      AvgPoolingFp16Grad(input_ptr + task_id * stride * out_batch_size, output_ptr + task_id * stride * in_batch_size,
                         count, pool_param, &compute_);
    }
  }
  return RET_OK;
}

int PoolingFp16GradImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto pooling = reinterpret_cast<PoolingGradCPUKernelFp16 *>(cdata);
  auto error_code = pooling->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pooling Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingGradCPUKernelFp16::Run() {
  thread_num_ = ms_context_->thread_num_;
  int error_code = ParallelLaunch(this->ms_context_, PoolingFp16GradImpl, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_AvgPoolGrad, LiteKernelCreator<PoolingGradCPUKernelFp16>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_MaxPoolGrad, LiteKernelCreator<PoolingGradCPUKernelFp16>)
}  // namespace mindspore::kernel
