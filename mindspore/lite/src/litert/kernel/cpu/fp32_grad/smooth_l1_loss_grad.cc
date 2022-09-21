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

#include "src/litert/kernel/cpu/fp32_grad/smooth_l1_loss_grad.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SmoothL1LossGrad;

namespace mindspore::kernel {
constexpr static int kPredictIdx = 0;
constexpr static int kTargetIdx = 1;
constexpr static int kDlossIdx = 2;
constexpr static int kOutputIdx = 0;

int SmoothL1LossGradCPUKernel::ReSize() {
  CHECK_NULL_RETURN(smooth_l1_param_);
  CHECK_NULL_RETURN(op_parameter_);
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_1D);
  CHECK_NULL_RETURN(in_tensors_.at(kPredictIdx));
  CHECK_NULL_RETURN(in_tensors_.at(kTargetIdx));
  CHECK_NULL_RETURN(in_tensors_.at(kDlossIdx));
  CHECK_NULL_RETURN(out_tensors_.at(kOutputIdx));
  return RET_OK;
}

int SmoothL1LossGradCPUKernel::DoExecute(int task_id) {
  SmoothL1LossParameter *smooth_l1_loss_param = reinterpret_cast<SmoothL1LossParameter *>(op_parameter_);
  auto predict = reinterpret_cast<float *>(in_tensors_.at(kPredictIdx)->MutableData());
  CHECK_NULL_RETURN(predict);
  auto target = reinterpret_cast<float *>(in_tensors_.at(kTargetIdx)->MutableData());
  CHECK_NULL_RETURN(target);
  auto d_loss = reinterpret_cast<float *>(in_tensors_.at(kDlossIdx)->MutableData());
  CHECK_NULL_RETURN(d_loss);
  auto *out = reinterpret_cast<float *>(out_tensors_.at(kOutputIdx)->MutableData());
  CHECK_NULL_RETURN(out);

  int length = in_tensors_.at(kPredictIdx)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;
  int end = start + count;

  const float beta = smooth_l1_loss_param->beta_;

  for (int i = start; i < end; ++i) {
    float diff = predict[i] - target[i];
    if (diff > beta) {
      out[i] = d_loss[i];
    } else if (diff < -beta) {
      out[i] = -d_loss[i];
    } else {
      out[i] = (diff / beta) * d_loss[i];
    }
  }
  return RET_OK;
}

int SmoothL1LossGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto smooth_l1_loss_kernel = reinterpret_cast<SmoothL1LossGradCPUKernel *>(cdata);
  CHECK_NULL_RETURN(smooth_l1_loss_kernel);
  auto error_code = smooth_l1_loss_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SmoothL1LossGrad error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SmoothL1LossGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, SmoothL1LossGradRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SmoothL1LossGrad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SmoothL1LossGradCPUKernel::Prepare() { return RET_OK; }

kernel::LiteKernel *CpuSmoothL1LossGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                         const std::vector<lite::Tensor *> &outputs,
                                                         OpParameter *opParameter, const lite::InnerContext *ctx,
                                                         const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_MSG(opParameter != nullptr, nullptr, "Op parameter is nullptr.");
  MS_ASSERT(desc.type == schema::PrimitiveType_SmoothL1LossGrad);
  auto *kernel = new (std::nothrow) SmoothL1LossGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SmoothL1LossGradWithLogitsCPUKernel failed";
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SmoothL1LossGrad, CpuSmoothL1LossGradFp32KernelCreator)
}  // namespace mindspore::kernel
