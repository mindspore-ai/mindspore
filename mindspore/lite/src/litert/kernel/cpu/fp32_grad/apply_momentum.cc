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

#include "src/litert/kernel/cpu/fp32_grad/apply_momentum.h"
#include <string>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ApplyMomentum;

namespace mindspore::kernel {
int ApplyMomentumCPUKernel::ReSize() { return RET_OK; }

static int DoApplyMomentum(float *weight, float *accumulate, float learning_rate, const float *gradient, float moment,
                           bool nesterov, int start, int end) {
  if (nesterov) {
    for (int i = start; i < end; i++) {
      accumulate[i] = accumulate[i] * moment + gradient[i];
      weight[i] -= (accumulate[i] * moment + gradient[i]) * learning_rate;
    }
  } else {
    for (int i = start; i < end; i++) {
      accumulate[i] = accumulate[i] * moment + gradient[i];
      weight[i] -= accumulate[i] * learning_rate;
    }
  }
  return RET_OK;
}

int ApplyMomentumCPUKernel::DoExecute(int task_id) {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_5D);
  auto weight = reinterpret_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
  CHECK_NULL_RETURN(weight);
  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
  CHECK_NULL_RETURN(accumulate);
  float learning_rate = lr_;
  auto gradient = reinterpret_cast<float *>(in_tensors_.at(FOURTH_INPUT)->data());
  CHECK_NULL_RETURN(gradient);
  CHECK_NULL_RETURN(in_tensors_.at(FIFTH_INPUT)->data());
  float moment = reinterpret_cast<float *>(in_tensors_.at(FIFTH_INPUT)->data())[0];
  int length = in_tensors_.at(FIRST_INPUT)->ElementsNum();

  MS_CHECK_TRUE_RET(thread_count_ > 0, RET_ERROR);
  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;
  int end = start + count;

  DoApplyMomentum(weight, accumulate, learning_rate, gradient, moment, apply_momentum_param_->use_nesterov_, start,
                  end);
  return RET_OK;
}

int ApplyMomentumRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto applyMomentum_kernel = reinterpret_cast<ApplyMomentumCPUKernel *>(cdata);
  auto error_code = RET_OK;
  if (applyMomentum_kernel->get_optimizer_mode() == WeightUpdateMode::VIRTUAL_BATCH) {
    error_code = applyMomentum_kernel->ExecuteVirtualBatch(task_id);
  } else if (applyMomentum_kernel->get_optimizer_mode() == WeightUpdateMode::ACCUMULATE_GRADS) {
    error_code = applyMomentum_kernel->ExecuteVirtualBatch(task_id);
  } else {
    error_code = applyMomentum_kernel->DoExecute(task_id);
  }
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "apply Momentum run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ApplyMomentumCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ApplyMomentumRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Apply Momentum function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ApplyMomentumCPUKernel::Prepare() {
  CHECK_NULL_RETURN(apply_momentum_param_);
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_5D);
  for (int i = 0; i < DIMENSION_5D; i++) {
    CHECK_NULL_RETURN(in_tensors_.at(i));
  }
  auto ret = OptimizerKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to initialize Apply Momentum Kernel";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<int> ApplyMomentumCPUKernel::GetOptimizerParamsIdxs() const {
  std::vector<int> indices = {4};
  return indices;
}

int ApplyMomentumCPUKernel::OptimizerStep() {
  auto weight = reinterpret_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
  CHECK_NULL_RETURN(weight);
  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
  CHECK_NULL_RETURN(accumulate);
  float learning_rate = lr_;
  CHECK_NULL_RETURN(in_tensors_.at(FIFTH_INPUT)->data());
  float moment = reinterpret_cast<float *>(in_tensors_.at(FIFTH_INPUT)->data())[0];

  size_t length = in_tensors_.at(FIRST_INPUT)->ElementsNum();

  if (grad_sum_ != nullptr && valid_grad_sum_) {
    size_t start = 0;
    size_t end = length;
    DoApplyMomentum(weight, accumulate, learning_rate, grad_sum_, moment, apply_momentum_param_->use_nesterov_, start,
                    end);
    std::fill(grad_sum_, grad_sum_ + length, 0);
    OptimizerKernel::OptimizerStep();
  }
  return RET_OK;
}

kernel::LiteKernel *CpuApplyMomentumFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::InnerContext *ctx,
                                                      const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_ApplyMomentum);
  auto *kernel = new (std::nothrow) ApplyMomentumCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ApplyMomentumCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ApplyMomentum, CpuApplyMomentumFp32KernelCreator)
}  // namespace mindspore::kernel
