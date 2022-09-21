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

#include "src/litert/kernel/cpu/fp32_grad/pooling_grad.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/pooling_fp32.h"
#include "nnacl/fp32_grad/pooling_grad.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AvgPoolGrad;
using mindspore::schema::PrimitiveType_MaxPoolGrad;

namespace mindspore::kernel {
namespace {
constexpr int kNumShapeDim2 = 1;
constexpr int kNumShapeDim3 = 2;
}  // namespace

int PoolingGradCPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(op_parameter_);
  CHECK_NULL_RETURN(pool_param);
  CHECK_NULL_RETURN(in_tensors_.at(FIRST_INPUT));
  CHECK_NULL_RETURN(in_tensors_.at(SECOND_INPUT));
  CHECK_NULL_RETURN(in_tensors_.at(THIRD_INPUT));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  auto in_shape = in_tensors_.at(0)->shape();
  auto out_shape = in_tensors_.at(1)->shape();
  MS_CHECK_TRUE_RET(in_shape.size() == COMM_SHAPE_SIZE, RET_ERROR);
  MS_CHECK_TRUE_RET(out_shape.size() == COMM_SHAPE_SIZE, RET_ERROR);

  if (pool_param->pool_mode_ == PoolMode_AvgPool) {
    out_shape = in_tensors_.at(THIRD_INPUT)->shape();
  }
  int input_h = in_shape.at(kNumShapeDim2);
  int input_w = in_shape.at(kNumShapeDim3);
  MS_CHECK_TRUE_RET(input_h > 0, RET_ERROR);
  MS_CHECK_TRUE_RET(input_w > 0, RET_ERROR);
  if (pool_param->global_) {
    pool_param->window_w_ = input_w;
    pool_param->window_h_ = input_h;
  }
  pool_param->input_h_ = in_shape[kNHWC_H];
  pool_param->input_w_ = in_shape[kNHWC_W];
  pool_param->input_batch_ = in_shape[kNHWC_N];
  pool_param->input_channel_ = in_shape[kNHWC_C];
  pool_param->output_h_ = out_shape[kNHWC_H];
  pool_param->output_w_ = out_shape[kNHWC_W];
  pool_param->output_batch_ = out_shape[kNHWC_N];
  pool_param->output_channel_ = out_shape[kNHWC_C];

  return RET_OK;
}

int PoolingGradCPUKernel::Prepare() { return ReSize(); }

int PoolingGradCPUKernel::DoExecute(int task_id) {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(op_parameter_);
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(output_ptr);
  MS_CHECK_TRUE_RET(thread_num_ > 0, RET_ERROR);
  int stride = UP_DIV(pool_param->output_batch_, thread_num_);
  int count = MSMIN(stride, pool_param->output_batch_ - stride * task_id);
  if (count > 0) {
    int in_batch_size = pool_param->input_h_ * pool_param->input_w_ * pool_param->input_channel_;
    int out_batch_size = pool_param->output_h_ * pool_param->output_w_ * pool_param->input_channel_;
    std::fill(output_ptr + task_id * stride * in_batch_size, output_ptr + ((task_id * stride) + count) * in_batch_size,
              0.f);
    if (pool_param->pool_mode_ == PoolMode_MaxPool) {
      auto dy_ptr = reinterpret_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
      MaxPoolingGrad(input_ptr + task_id * stride * in_batch_size, dy_ptr + task_id * stride * out_batch_size,
                     output_ptr + task_id * stride * in_batch_size, count, pool_param);
    } else {
      input_ptr = reinterpret_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
      AvgPoolingGrad(input_ptr + task_id * stride * out_batch_size, output_ptr + task_id * stride * in_batch_size,
                     count, pool_param);
    }
  }
  return RET_OK;
}

int PoolingGradImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto pooling = reinterpret_cast<PoolingGradCPUKernel *>(cdata);
  auto error_code = pooling->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pooling Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingGradCPUKernel::Run() {
  thread_num_ = op_parameter_->thread_num_;
  int error_code = ParallelLaunch(this->ms_context_, PoolingGradImpl, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPoolingGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                    const std::vector<lite::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::InnerContext *ctx,
                                                    const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  auto *kernel = new (std::nothrow) PoolingGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PoolingGradCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AvgPoolGrad, CpuPoolingGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MaxPoolGrad, CpuPoolingGradFp32KernelCreator)
}  // namespace mindspore::kernel
