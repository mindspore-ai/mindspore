/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32_grad/adam_weight_decay.h"
#include <cmath>
#include <string>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AdamWeightDecay;

namespace mindspore::kernel {
namespace {
constexpr static int kBeta1Idx = 4;
constexpr static int kBeta2Idx = 5;
constexpr static int kEpsilonIdx = 6;
constexpr static int kDecayIdx = 7;
}  // namespace

int AdamWeightDecayCPUKernel::ReSize() { return RET_OK; }

int AdamWeightDecayCPUKernel::DoExecute(int task_id) {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_9D);
  auto weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIdx)->MutableData());
  auto m = reinterpret_cast<float *>(in_tensors_.at(kMomentVector1stIdx)->MutableData());
  auto v = reinterpret_cast<float *>(in_tensors_.at(kMomentVector2stIdx)->MutableData());
  auto learning_rate = lr_;
  auto beta1 = reinterpret_cast<float *>(in_tensors_.at(kBeta1Idx)->MutableData())[0];
  auto beta2 = reinterpret_cast<float *>(in_tensors_.at(kBeta2Idx)->MutableData())[0];
  auto eps = reinterpret_cast<float *>(in_tensors_.at(kEpsilonIdx)->MutableData())[0];
  auto decay = reinterpret_cast<float *>(in_tensors_.at(kDecayIdx)->MutableData())[0];
  auto gradient = reinterpret_cast<float *>(in_tensors_.at(kGradientIdx)->MutableData());
  int length = in_tensors_.at(kWeightIdx)->ElementsNum();
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(m);
  CHECK_NULL_RETURN(v);
  CHECK_NULL_RETURN(gradient);

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  int start = stride * task_id;
  int end = start + count;

  return AdamWeightDecayFp32(weight, m, v, learning_rate, beta1, beta2, eps, decay, gradient, start, end);
}

int AdamWeightDecayRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto adam_weight_decay_kernel = reinterpret_cast<AdamWeightDecayCPUKernel *>(cdata);
  CHECK_NULL_RETURN(adam_weight_decay_kernel);
  auto error_code = RET_OK;
  if (adam_weight_decay_kernel->get_optimizer_mode() == WeightUpdateMode::VIRTUAL_BATCH) {
    error_code = adam_weight_decay_kernel->ExecuteVirtualBatch(task_id);
  } else if (adam_weight_decay_kernel->get_optimizer_mode() == WeightUpdateMode::ACCUMULATE_GRADS) {
    error_code = adam_weight_decay_kernel->ExecuteVirtualBatch(task_id);
  } else {
    error_code = adam_weight_decay_kernel->DoExecute(task_id);
  }

  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "AdamWeightDecay run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AdamWeightDecayCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, AdamWeightDecayRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "AdamWeightDecay function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AdamWeightDecayCPUKernel::Prepare() {
  auto ret = OptimizerKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to initialize AdamWeightDecay Kernel";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<int> AdamWeightDecayCPUKernel::GetOptimizerParamsIdxs() const {
  std::vector<int> indices = {4, 5, 6, 7};
  return indices;
}

int AdamWeightDecayCPUKernel::OptimizerStep() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_9D - 1);
  auto weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIdx)->MutableData());
  auto m = reinterpret_cast<float *>(in_tensors_.at(kMomentVector1stIdx)->MutableData());
  auto v = reinterpret_cast<float *>(in_tensors_.at(kMomentVector2stIdx)->MutableData());
  auto learning_rate = lr_;
  auto beta1 = reinterpret_cast<float *>(in_tensors_.at(kBeta1Idx)->MutableData())[0];
  auto beta2 = reinterpret_cast<float *>(in_tensors_.at(kBeta2Idx)->MutableData())[0];
  auto eps = reinterpret_cast<float *>(in_tensors_.at(kEpsilonIdx)->MutableData())[0];
  auto decay = reinterpret_cast<float *>(in_tensors_.at(kDecayIdx)->MutableData())[0];
  int length = in_tensors_.at(kWeightIdx)->ElementsNum();
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(m);
  CHECK_NULL_RETURN(v);

  int ret = RET_OK;
  if (grad_sum_ != nullptr && valid_grad_sum_) {
    size_t start = 0;
    size_t end = length;
    ret = AdamWeightDecayFp32(weight, m, v, learning_rate, beta1, beta2, eps, decay, grad_sum_, start, end);
    std::fill(grad_sum_, grad_sum_ + length, 0);
    OptimizerKernel::OptimizerStep();
  }
  return ret;
}

kernel::LiteKernel *CpuAdamWeightDecayFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                        const std::vector<lite::Tensor *> &outputs,
                                                        OpParameter *opParameter, const lite::InnerContext *ctx,
                                                        const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_MSG(opParameter != nullptr, nullptr, "Op parameter is nullptr.");
  MS_ASSERT(desc.type == schema::PrimitiveType_AdamWeightDecay);
  auto *kernel = new (std::nothrow) AdamWeightDecayCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new AdamWeightDecayCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AdamWeightDecay, CpuAdamWeightDecayFp32KernelCreator)
}  // namespace mindspore::kernel
